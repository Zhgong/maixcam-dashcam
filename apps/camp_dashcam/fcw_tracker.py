"""
Core FCW (Forward Collision Warning) Analyzer & Monocular Tracking Engine
Supports Virtual Lane Corridor (Neighboring Lane Suppression) & Alert Debounce (v1.2.0)
"""

import math
try:
    from config import DashcamConfig
except ImportError:
    from .config import DashcamConfig



class AlertLevel:
    NONE = "NONE"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class GravityDirection:
    DOWN = "DOWN"        # 重力指向传感器 +Y 方向 (正常直立正装)
    UP = "UP"            # 重力指向传感器 -Y 方向 (倒吊吸盘倒装)
    UNKNOWN = "UNKNOWN"


class BottomEdge:
    IMAGE_BOTTOM = "IMAGE_BOTTOM"  # 物理底部对应图像底部 (y = H)
    IMAGE_TOP = "IMAGE_TOP"        # 物理底部对应图像顶部 (y = 0)


class OrientationDetector:
    """
    负责基于 3 轴重力加速度实时识别重力方向并确定画面物理底部。
    具备启动零值过滤、EMA滑动滤波与迟滞防抖区间。
    """
    def __init__(self, ema_alpha: float = 0.25, threshold: float = 2.5, config=DashcamConfig):
        self.config = config
        self.ema_alpha = ema_alpha
        self.threshold = threshold
        self.smooth_ay = 9.8  # 默认正常直立正放
        self.is_inverted = False
        self.gravity_direction = GravityDirection.DOWN
        self.bottom_edge = BottomEdge.IMAGE_BOTTOM
        self.has_valid_sample = False

        # 初始化时读取强制安装模式配置
        mount_mode = getattr(self.config, "MOUNT_MODE", "auto")
        if mount_mode == "inverted":
            self.is_inverted = True
            self.gravity_direction = GravityDirection.UP
            self.bottom_edge = BottomEdge.IMAGE_TOP
            self.smooth_ay = -9.8
        elif mount_mode == "upright":
            self.is_inverted = False
            self.gravity_direction = GravityDirection.DOWN
            self.bottom_edge = BottomEdge.IMAGE_BOTTOM
            self.smooth_ay = 9.8

    def update(self, acc_x: float, acc_y: float, acc_z: float) -> bool:
        """
        输入实时加速度计数据，更新重力方向与底部判定。
        :return: bool, 朝向是否发生切换 (changed)
        """
        mount_mode = getattr(self.config, "MOUNT_MODE", "auto")
        if mount_mode == "inverted":
            old_inv = self.is_inverted
            self.is_inverted = True
            self.gravity_direction = GravityDirection.UP
            self.bottom_edge = BottomEdge.IMAGE_TOP
            return old_inv != self.is_inverted
        elif mount_mode == "upright":
            old_inv = self.is_inverted
            self.is_inverted = False
            self.gravity_direction = GravityDirection.DOWN
            self.bottom_edge = BottomEdge.IMAGE_BOTTOM
            return old_inv != self.is_inverted

        # auto 模式: 基于 3 轴重力加速度实时识别重力方向并确定画面物理底部
        # 1. 过滤无效/未就绪的零值数据或失重状态
        norm_sq = acc_x**2 + acc_y**2 + acc_z**2
        if norm_sq < 10.0:  # < ~1G 的 1/3，视作传感器未稳定或自由落体
            return False

        if not self.has_valid_sample:
            self.smooth_ay = acc_y
            self.has_valid_sample = True
        else:
            self.smooth_ay = (1.0 - self.ema_alpha) * self.smooth_ay + self.ema_alpha * acc_y

        old_inv = self.is_inverted

        # 2. 基于迟滞阈值识别重力方向并确定物理底部
        if self.smooth_ay > self.threshold:
            # 相机处于正放姿态 (acc_y > 2.5 m/s²)，原生画面为正向，物理底部即为图像底部，无需翻转
            self.is_inverted = False
            self.gravity_direction = GravityDirection.DOWN
            self.bottom_edge = BottomEdge.IMAGE_BOTTOM
        elif self.smooth_ay < -self.threshold:
            # 相机处于倒放吊装姿态 (acc_y < -2.5 m/s²)，物理底部对应图像顶部，触发 HUD 翻转保持正向
            self.is_inverted = True
            self.gravity_direction = GravityDirection.UP
            self.bottom_edge = BottomEdge.IMAGE_TOP
        # 在 [-threshold, +threshold] 之间保持当前状态 (防抖与防翻转抖动)

        return old_inv != self.is_inverted


class FCWTarget:
    def __init__(self, track_id: int, class_id: int, bbox: list, distance_m: float, timestamp: float):
        self.track_id = track_id
        self.class_id = class_id
        self.bbox = bbox  # [x, y, w, h]
        self.distance_m = distance_m
        self.last_distance = distance_m
        self.last_time = timestamp
        self.rel_speed_mps = 0.0     # 相对逼近速度 (m/s), 正数表示前车逼近/本车靠近
        self.ttc_sec = float("inf")  # 碰撞时间 (Time-To-Collision)
        self.lateral_x_m = 0.0       # 物理横向距离 (m), 0 为正中, 负为偏左, 正为偏右
        self.last_lateral_x = 0.0
        self.lateral_v_mps = 0.0     # 横向漂移速度 (m/s)
        self.in_lane_corridor = True # 是否位于本车道走廊范围 (|X| <= 1.35m)
        self.confirm_counter = 0     # 预警连续确认帧数计数器 (防抖机制)
        self.active_alert_level = AlertLevel.NONE # 经过防抖确认后的生效告警级别
        self.lost_frames = 0

    def update(self, bbox: list, distance_m: float, lateral_x_m: float, timestamp: float, config=DashcamConfig, yaw_rate_dps: float = 0.0):
        # 1. EMA 指数滑动平均滤波 (抑制路面颠簸引起的单帧像素抖动)
        alpha = getattr(config, "DISTANCE_SMOOTH_ALPHA", 0.5)
        smooth_dist = alpha * distance_m + (1.0 - alpha) * self.distance_m

        dt = timestamp - self.last_time
        if dt >= config.MIN_DT_SEC:
            # 相对纵向逼近速度 = (前一平滑距离 - 当前平滑距离) / 时间差
            v_rel = (self.last_distance - smooth_dist) / dt
            self.rel_speed_mps = v_rel
            self.last_distance = smooth_dist

            # 横向切向漂移速度 = (当前横向位移 - 前一横向位移) / 时间差
            prev_x = self.last_lateral_x if self.last_lateral_x != 0.0 else self.lateral_x_m
            self.lateral_v_mps = (lateral_x_m - prev_x) / dt
            self.last_lateral_x = lateral_x_m

            self.last_time = timestamp

            # 仅在相对逼近速度大于阈值时计算有限 TTC，否则视为静止/远离 (TTC = inf)
            if v_rel >= config.MIN_APPROACH_SPEED_MPS and smooth_dist > 0:
                self.ttc_sec = smooth_dist / v_rel
            else:
                self.ttc_sec = float("inf")
        else:
            self.last_lateral_x = lateral_x_m

        self.bbox = bbox
        self.distance_m = smooth_dist
        self.lateral_x_m = lateral_x_m

        # 动态弯道走廊判定 (根据偏航角速度与距离推算走廊中心偏移)
        omega_rad = math.radians(yaw_rate_dps)
        v_est = getattr(config, "ESTIMATED_SPEED_MPS", 10.0)
        max_kappa = getattr(config, "MAX_CURVATURE_INV_M", 0.06)
        kappa = max(-max_kappa, min(max_kappa, omega_rad / v_est))
        center_x = -0.5 * kappa * (smooth_dist ** 2) if abs(yaw_rate_dps) >= 0.5 else 0.0
        corridor_half_w = getattr(config, "LANE_CORRIDOR_HALF_WIDTH_M", 1.35)
        self.in_lane_corridor = abs(lateral_x_m - center_x) <= corridor_half_w
        self.lost_frames = 0



class FCWAnalyzer:
    def __init__(self, config=DashcamConfig):
        self.config = config
        self.tracks = {}  # {track_id: FCWTarget}

        # 几何参数预计算
        self.cam_height = config.CAMERA_HEIGHT_M
        self.pitch_rad = math.radians(config.CAMERA_PITCH_DEG)
        self.fov_v_rad = math.radians(config.CAMERA_FOV_V_DEG)
        self.fov_h_rad = math.radians(getattr(config, "CAMERA_FOV_H_DEG", 55.0))
        self.img_w = config.IMG_WIDTH
        self.img_h = config.IMG_HEIGHT
        self.x_center = self.img_w / 2.0
        self.y_center = self.img_h / 2.0

        # 焦距计算 (像素单位)
        # fy = (H / 2) / tan(FOV_v / 2)
        # fx = (W / 2) / tan(FOV_h / 2)
        self.fy = self.y_center / math.tan(self.fov_v_rad / 2.0)
        self.fx = self.x_center / math.tan(self.fov_h_rad / 2.0)
        self.calibrated_pitch_deg = config.CAMERA_PITCH_DEG
        self.current_pitch_deg = config.CAMERA_PITCH_DEG
        self.current_horizon_y = self.compute_horizon_y(self.current_pitch_deg)

    @staticmethod
    def is_camera_inverted(acc_y: float) -> bool:
        """
        根据静态重力加速度 Y 轴分量判断相机安装朝向:
        - acc_y > 3.0 m/s²: 正放 (is_inverted = False)
        - acc_y < -3.0 m/s²: 倒放吊装 (is_inverted = True)
        """
        return acc_y < -3.0

    @staticmethod
    def compute_pitch_from_gravity(acc_x: float, acc_y: float, acc_z: float) -> float:
        """
        利用静止/巡航时 3 轴重力矢量解算相机安装俯仰角 (Pitch Angle, 度):
        平视水平时 acc_z ≈ 0.0, acc_y ≈ 9.8 -> pitch = 0.0°
        仰头安装时 (镜头朝向天空)，重力向后倒向 -Z，因此 acc_z < 0 -> pitch < 0° (如 -19.2°)
        """
        xy_norm = math.sqrt(acc_x ** 2 + acc_y ** 2)
        if xy_norm < 1e-3:
            return 0.0
        pitch_rad = math.atan2(acc_z, xy_norm)
        return float(math.degrees(pitch_rad))

    def update_pitch_with_gravity(self, acc_x: float, acc_y: float, acc_z: float) -> float:
        """
        [Slope & Brake Immunity Gate]:
        相机吸盘固定在挡风玻璃后，物理安装角保持恒定不变。
        利用低通滤波（长半衰期）仅标定恒定的物理安装仰角，
        彻底免疫车辆上下坡引起的重力倾斜与急刹车惯性力震荡！
        """
        raw_pitch = self.compute_pitch_from_gravity(acc_x, acc_y, acc_z)
        if self.calibrated_pitch_deg == 0.0:
            # 启动首次采样快速对齐真实安装角
            self.calibrated_pitch_deg = raw_pitch
        else:
            # 极低学习率长平滑: 0.98 * 旧值 + 0.02 * 新值
            self.calibrated_pitch_deg = 0.98 * self.calibrated_pitch_deg + 0.02 * raw_pitch

        self.current_pitch_deg = self.calibrated_pitch_deg
        self.current_horizon_y = self.compute_horizon_y(self.current_pitch_deg)
        return self.current_pitch_deg

    def compute_horizon_y(self, pitch_deg: float = None) -> float:
        """
        根据相机俯仰角 (pitch_deg) 动态计算图像上的真实地平线纵坐标 y_horizon (像素):
        y_horizon = c_y - f_y * tan(pitch_rad)
        - 平视 (pitch=0°): 地平线位于画面正中 (y = c_y = 240px)
        - 仰头 (pitch < 0°): 地平线在画面中下移 (y > 240px，如 300px)
        """
        p_deg = pitch_deg if pitch_deg is not None else self.current_pitch_deg
        p_rad = math.radians(p_deg)
        h_y = self.y_center - self.fy * math.tan(p_rad)
        return float(h_y)


    def compute_corridor_center_offset(self, distance_z: float, yaw_rate_dps: float = 0.0) -> float:
        """
        [Dynamic Curvature Kinematics]:
        根据 IMU 偏航角速度 (yaw_rate_dps) 与阿克曼转向几何计算距离 Z 处的动态弯道车道中心横向偏移量 Delta X (米)。
        - yaw_rate > 0 (左转): 弯道向左延伸，Delta X < 0
        - yaw_rate < 0 (右转): 弯道向右延伸，Delta X > 0
        Delta X(Z) = -0.5 * kappa * (Z ** 2)
        """
        if abs(yaw_rate_dps) < 0.5 or distance_z <= 0.5:
            return 0.0

        omega_rad = math.radians(yaw_rate_dps)
        v_est = getattr(self.config, "ESTIMATED_SPEED_MPS", 10.0)
        max_kappa = getattr(self.config, "MAX_CURVATURE_INV_M", 0.06)
        kappa = max(-max_kappa, min(max_kappa, omega_rad / v_est))
        offset = -0.5 * kappa * (distance_z ** 2)
        return float(offset)

    def is_in_corridor(self, lateral_x: float, distance_z: float, yaw_rate_dps: float = 0.0) -> bool:
        """
        判断目标是否位于本车动态弯曲走廊内 (|X - X_center(Z)| <= CorridorHalfWidth)
        """
        center_x = self.compute_corridor_center_offset(distance_z, yaw_rate_dps)
        corridor_half_w = getattr(self.config, "LANE_CORRIDOR_HALF_WIDTH_M", 1.35)
        return abs(lateral_x - center_x) <= corridor_half_w

    def estimate_distance(self, bottom_y: float, pitch_deg: float = None) -> float:
        """
        根据检测框底边 y 坐标估算目标的实际物理纵向距离 Z (米)。
        支持传入动态俯仰角或默认静态角。
        """
        p_deg = pitch_deg if pitch_deg is not None else self.current_pitch_deg
        p_rad = math.radians(p_deg)
        horizon_y = self.compute_horizon_y(p_deg)

        if bottom_y <= horizon_y:
            # 接地点在真实地平线之上或天空，超出单目路面几何模型范围
            return self.config.MAX_EVAL_DISTANCE_M * 2

        dy = bottom_y - self.y_center
        alpha = math.atan(dy / self.fy)
        total_angle = p_rad + alpha

        if total_angle <= 0.02:  # 趋于平行地平线
            return self.config.MAX_EVAL_DISTANCE_M * 2

        distance = self.cam_height / math.tan(total_angle)
        return float(distance)


    def estimate_lateral_x(self, bbox_center_x: float, distance_z: float) -> float:
        """
        [Virtual Lane Corridor]:
        根据框中心横向像素与纵向距离计算目标的物理横向坐标 X (米)。
        X = (x_center - cx) * Z / fx
        0 表示车道正中，负数表示偏左，正数表示偏右。
        """
        dx_pixels = bbox_center_x - self.x_center
        x_meters = (dx_pixels * distance_z) / self.fx
        return float(x_meters)

    def is_valid_detection(self, det: dict) -> bool:
        """
        过滤非关注目标类别与主车道扇区 (ROI) 外的目标。
        """
        class_id = det.get("class_id")
        if class_id not in self.config.VALID_CLASSES:
            return False


        bbox = det.get("bbox", [0, 0, 0, 0])
        x, y, w, h = bbox
        x_center = x + w / 2.0
        bottom_y = y + h

        # 归一化坐标检查
        x_norm = x_center / self.img_w
        y_norm = bottom_y / self.img_h

        if x_norm < self.config.ROI_X_MIN_RATIO or x_norm > self.config.ROI_X_MAX_RATIO:
            return False
        if y_norm < self.config.ROI_Y_MIN_RATIO:
            return False

        return True

    def evaluate_raw_alert_level(self, target: FCWTarget) -> str:
        """
        根据距离、TTC、虚拟车道走廊与横向漂移切出过滤评估单帧原始预警级别。
        """
        # 1. 如果超出最大评估距离或位于本车道走廊外 (|X| > 1.35m)，抑制所有警报！
        if target.distance_m > self.config.MAX_EVAL_DISTANCE_M or not target.in_lane_corridor:
            return AlertLevel.NONE

        # 2. [Lateral Drift / Cut-out Gate]: 横向切出与转弯漂移抑制
        # 当车辆转弯或本车在窄道上掠过路边停放车辆时，目标在画面上会产生显著的横向平移 (|vx| > 1.2 m/s)。
        # 如果目标偏离车道中心线 (|X| > 0.8m) 且横向横移速度过大，说明不是本车道迎头碰撞对象，抑制报警！
        max_drift = getattr(self.config, "MAX_LATERAL_DRIFT_SPEED_MPS", 1.2)
        if abs(target.lateral_v_mps) > max_drift and abs(target.lateral_x_m) > 0.8:
            return AlertLevel.NONE

        # 3. 评估紧急制动警报 (CRITICAL)
        if target.ttc_sec < self.config.CRITICAL_TTC_SEC:
            return AlertLevel.CRITICAL

        # 4. 评估注意预警 (WARNING)
        if target.ttc_sec < self.config.WARNING_TTC_SEC and target.distance_m <= self.config.WARNING_DISTANCE_M:
            return AlertLevel.WARNING

        return AlertLevel.NONE

    def evaluate_debounced_alert_level(self, target: FCWTarget, raw_level: str) -> str:
        """
        [Alert Debounce]: 3 帧连续确认防抖状态机。
        必须连续达到阈值 DEBOUNCE_CONFIRM_FRAMES 次才正式生效，
        过滤单帧颠簸毛刺，消除频闪红框。
        """
        required_frames = getattr(self.config, "DEBOUNCE_CONFIRM_FRAMES", 3)

        if raw_level == AlertLevel.NONE:
            target.confirm_counter = 0
            target.active_alert_level = AlertLevel.NONE
            return AlertLevel.NONE

        if raw_level == AlertLevel.CRITICAL:
            target.confirm_counter += 1
            if target.confirm_counter >= required_frames:
                target.active_alert_level = AlertLevel.CRITICAL
                return AlertLevel.CRITICAL
            return AlertLevel.NONE

        if raw_level == AlertLevel.WARNING:
            target.confirm_counter += 1
            if target.confirm_counter >= required_frames:
                target.active_alert_level = AlertLevel.WARNING
                return AlertLevel.WARNING
            return AlertLevel.NONE

        return AlertLevel.NONE

    def evaluate_alert_level(self, target: FCWTarget) -> str:
        """
        兼容旧接口直接评估
        """
        raw = self.evaluate_raw_alert_level(target)
        return self.evaluate_debounced_alert_level(target, raw)

    def process_detections(self, detections: list, now_sec: float, yaw_rate_dps: float = 0.0) -> list:
        """
        处理当前帧检测结果，更新追踪目标并返回告警列表。
        :param detections: list of dict {'track_id': int, 'class_id': int, 'bbox': [x,y,w,h]}
        :param now_sec: 当前时间戳 (秒)
        :param yaw_rate_dps: 偏航角速度 (deg/s), 动态弯道走廊补偿
        :return: list of dict [{'level': AlertLevel, 'target': FCWTarget}]
        """
        current_seen_ids = set()
        alerts = []

        # 1. 过滤并更新/注册可见目标
        for det in detections:
            if not self.is_valid_detection(det):
                continue

            t_id = det["track_id"]
            class_id = det["class_id"]
            bbox = det["bbox"]
            x, y, w, h = bbox
            bbox_center_x = x + w / 2.0
            bottom_y = y + h
            distance = self.estimate_distance(bottom_y)
            lateral_x = self.estimate_lateral_x(bbox_center_x, distance)

            current_seen_ids.add(t_id)

            if t_id not in self.tracks:
                self.tracks[t_id] = FCWTarget(
                    track_id=t_id,
                    class_id=class_id,
                    bbox=bbox,
                    distance_m=distance,
                    timestamp=now_sec
                )
                self.tracks[t_id].lateral_x_m = lateral_x
                self.tracks[t_id].last_lateral_x = lateral_x
                self.tracks[t_id].in_lane_corridor = self.is_in_corridor(lateral_x, distance, yaw_rate_dps)
            else:
                self.tracks[t_id].update(
                    bbox=bbox,
                    distance_m=distance,
                    lateral_x_m=lateral_x,
                    timestamp=now_sec,
                    config=self.config,
                    yaw_rate_dps=yaw_rate_dps
                )

            target = self.tracks[t_id]
            raw_level = self.evaluate_raw_alert_level(target)
            active_level = self.evaluate_debounced_alert_level(target, raw_level)
            if active_level != AlertLevel.NONE:
                alerts.append({"level": active_level, "target": target})

        # 2. 目标生命周期管理与丢失帧清理
        stale_ids = []
        for t_id, target in self.tracks.items():
            if t_id not in current_seen_ids:
                target.lost_frames += 1
                if target.lost_frames >= self.config.TRACK_MAX_LOST_FRAMES:
                    stale_ids.append(t_id)

        for t_id in stale_ids:
            del self.tracks[t_id]

        return alerts
