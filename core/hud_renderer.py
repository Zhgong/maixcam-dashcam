"""
HUD Renderer for Dashcam: Real-time ADAS Overlay, Distance Badges & Alert Banners
"""

import time
import math
try:
    from config import DashcamConfig
    from fcw_tracker import AlertLevel, FCWTarget
except ImportError:
    from .config import DashcamConfig
    from .fcw_tracker import AlertLevel, FCWTarget



def get_color(r, g, b):
    try:
        from maix import image
        return image.Color.from_rgb(r, g, b)
    except Exception:
        return (r, g, b)


class HUDRenderer:
    # 颜色常量定义 (兼容 MaixPy image.Color 与 Python tuple)
    COLOR_SAFE = get_color(46, 204, 113)      # 翠绿 (安全)
    COLOR_WARNING = get_color(241, 196, 15)   # 亮黄 (注意)
    COLOR_CRITICAL = get_color(231, 76, 60)   # 鲜红 (紧急制动)
    COLOR_CYAN = get_color(52, 152, 219)      # 科技蓝 (ROI与状态)
    COLOR_ROAD = get_color(243, 156, 18)      # 琥珀金 (实测物理道路边缘)
    COLOR_HORIZON = get_color(127, 140, 141)  # 灰蓝 (自标定地平线)
    COLOR_WHITE = get_color(255, 255, 255)
    COLOR_DARK_BG = get_color(20, 24, 33)


    CLASS_NAMES = {
        DashcamConfig.TARGET_PERSON: "Person",
        DashcamConfig.TARGET_BICYCLE: "Bike",
        DashcamConfig.TARGET_CAR: "Car",
        DashcamConfig.TARGET_MOTORCYCLE: "Motor",
        DashcamConfig.TARGET_BUS: "Bus",
        DashcamConfig.TARGET_TRUCK: "Truck"
    }

    def __init__(self, config=DashcamConfig):
        self.config = config
        self.cam_h = getattr(config, "CAMERA_HEIGHT_M", 1.3)
        self.pitch_rad = math.radians(getattr(config, "CAMERA_PITCH_DEG", 0.0))
        self.fov_v_rad = math.radians(getattr(config, "CAMERA_FOV_V_DEG", 45.0))
        self.fov_h_rad = math.radians(getattr(config, "CAMERA_FOV_H_DEG", 55.0))
        self.img_w = config.IMG_WIDTH
        self.img_h = config.IMG_HEIGHT
        self.cx = self.img_w / 2.0
        self.cy = self.img_h / 2.0
        self.fy = self.cy / math.tan(self.fov_v_rad / 2.0)
        self.fx = self.cx / math.tan(self.fov_h_rad / 2.0)
        self.corridor_half_w = getattr(config, "LANE_CORRIDOR_HALF_WIDTH_M", 1.35)
        self.z_near = getattr(config, "CORRIDOR_Z_NEAR_M", 3.5)
        self.z_far = getattr(config, "CORRIDOR_Z_FAR_M", 25.0)

    def render_actual_road(self, img, road_res):
        """
        绘制从实际路面图像中提取出的物理车道标线与沥青路缘多段线 (Real Road Boundaries)。
        """
        if not road_res or not getattr(road_res, "detected", False):
            return

        left_pts = getattr(road_res, "left_boundary_pts", [])
        right_pts = getattr(road_res, "right_boundary_pts", [])
        road_color = self.COLOR_ROAD

        # 1. 绘制左侧物理边界线
        if left_pts and len(left_pts) >= 2:
            for i in range(len(left_pts) - 1):
                p1, p2 = left_pts[i], left_pts[i + 1]
                img.draw_line(p1[0], p1[1], p2[0], p2[1], color=road_color, thickness=3)

        # 2. 绘制右侧物理边界线
        if right_pts and len(right_pts) >= 2:
            for i in range(len(right_pts) - 1):
                p1, p2 = right_pts[i], right_pts[i + 1]
                img.draw_line(p1[0], p1[1], p2[0], p2[1], color=road_color, thickness=3)

        # 3. 标识当前道路感知状态徽章
        if left_pts and right_pts:
            tag_x = int(left_pts[0][0] + (right_pts[0][0] - left_pts[0][0]) / 2.0 - 50)
            tag_y = min(self.img_h - 35, max(left_pts[0][1], right_pts[0][1]) - 15)
            road_type_tag = "LANE" if getattr(road_res, "road_type", "") == "painted_lanes" else "ROAD"
            conf_pct = int(road_res.confidence * 100)
            img.draw_string(max(10, tag_x), tag_y, f"[{road_type_tag} {conf_pct}%]", color=road_color, scale=0.8)

    def project_ground_point(self, x_m: float, z_m: float, yaw_rate_dps: float = 0.0):
        """将地面物理坐标 (X_m, Z_m) 透视投影至屏幕像素坐标 (px, py)，支持动态转弯曲率偏移"""
        if z_m <= 0.1:
            return int(self.cx), int(self.cy)

        # 动态弯道曲率横向偏移 (左转 offset < 0，右转 offset > 0)
        curve_offset = 0.0
        if abs(yaw_rate_dps) >= 0.5:
            omega_rad = math.radians(yaw_rate_dps)
            v_est = getattr(self.config, "ESTIMATED_SPEED_MPS", 10.0)
            max_kappa = getattr(self.config, "MAX_CURVATURE_INV_M", 0.06)
            kappa = max(-max_kappa, min(max_kappa, omega_rad / v_est))
            curve_offset = -0.5 * kappa * (z_m ** 2)

        effective_x = x_m + curve_offset

        total_angle = math.atan(self.cam_h / z_m)
        alpha = total_angle - self.pitch_rad
        dy = self.fy * math.tan(alpha)
        py = int(self.cy + dy)
        dx = (effective_x * self.fx) / z_m
        px = int(self.cx + dx)
        px = max(-200, min(self.img_w + 200, px))
        py = max(0, min(self.img_h - 1, py))
        return px, py

    def render_lane_corridor(self, img, alerts: list, yaw_rate_dps: float = 0.0):
        """
        绘制虚拟车道透视安全地毯包络线 (Virtual Lane Carpet)。
        支持基于偏航角速度 (yaw_rate_dps) 的平滑动态转弯随动弧线。
        并在 5m, 10m, 20m 处绘制安全距离刻度横梁。
        """
        # 判断当前是否有碰撞告警，动态切换地毯高亮颜色
        if any(a["level"] == AlertLevel.CRITICAL for a in alerts):
            line_color = self.COLOR_CRITICAL
        elif any(a["level"] == AlertLevel.WARNING for a in alerts):
            line_color = self.COLOR_WARNING
        else:
            line_color = self.COLOR_CYAN

        w_half = self.corridor_half_w

        # 动态弯道采用多段线采样拟合平滑弧线
        z_steps = [self.z_near, 5.0, 7.5, 11.0, 16.0, 21.0, self.z_far]
        pts_left = [self.project_ground_point(-w_half, z, yaw_rate_dps) for z in z_steps]
        pts_right = [self.project_ground_point(w_half, z, yaw_rate_dps) for z in z_steps]

        # 1. 绘制左右分段多段线
        for i in range(len(z_steps) - 1):
            img.draw_line(pts_left[i][0], pts_left[i][1], pts_left[i+1][0], pts_left[i+1][1], color=line_color, thickness=2)
            img.draw_line(pts_right[i][0], pts_right[i][1], pts_right[i+1][0], pts_right[i+1][1], color=line_color, thickness=2)

        # 2. 绘制安全距离梯级横梁 (5m, 10m, 20m)
        for dist_tick in [5.0, 10.0, 20.0]:
            if self.z_near <= dist_tick <= self.z_far:
                lx, ly = self.project_ground_point(-w_half, dist_tick, yaw_rate_dps)
                rx, ry = self.project_ground_point(w_half, dist_tick, yaw_rate_dps)
                img.draw_line(lx, ly, rx, ry, color=line_color, thickness=1)
                img.draw_string(rx + 4, ly - 7, f"{int(dist_tick)}m", color=line_color, scale=0.7)

    @staticmethod
    def format_duration(seconds: int) -> str:
        """将秒数格式化为 MM:SS"""
        m = seconds // 60
        s = seconds % 60
        return f"{m:02d}:{s:02d}"

    def draw_rotated_bar(self, img, bar_w: int, bar_h: int, dest_x: int, dest_y: int,
                         bg_color, draw_content_fn, is_inverted: bool):
        """
        在画面指定区域绘制条带内容。当处于倒装模式 (is_inverted=True) 时，
        在条带缓冲画布上绘制正向文字后整体旋转 180° 贴回图像，
        确保在物理倒立的屏幕上，驾驶员肉眼看到的文字 100% 正立、无任何镜像颠倒！
        """
        try:
            from maix import image as m_image
            has_maix_image = hasattr(m_image, "Image") and hasattr(img, "draw_image")
        except Exception:
            has_maix_image = False

        if not is_inverted or not has_maix_image:
            img.draw_rect(dest_x, dest_y, bar_w, bar_h, color=bg_color, thickness=-1)
            draw_content_fn(img, dest_x, dest_y)
        else:
            bar = m_image.Image(bar_w, bar_h, m_image.Format.FMT_RGB888)
            bar.draw_rect(0, 0, bar_w, bar_h, color=bg_color, thickness=-1)
            draw_content_fn(bar, 0, 0)
            bar_rot = bar.rotate(180)
            img.draw_image(dest_x, dest_y, bar_rot)

    def render_hud(self, img, tracks: dict, alerts: list, is_recording: bool = True,
                   is_locked: bool = False, rec_seconds: int = 0, temp_c: int = 45,
                   total_g: float = 1.0, snap_count: int = 0, yaw_rate_dps: float = 0.0,
                   road_res = None, horizon_y: int = None, acc_y: float = 9.8,
                   gravity_direction: str = "DOWN", is_inverted: bool = False):
        """
        在画面上绘制高对比度暗黑 ADAS HUD。
        :param img: maix.image.Image 或 Mock 绘图对象
        :param tracks: dict of {track_id: FCWTarget}
        :param alerts: list of alert dicts
        :param is_recording: 是否正在录像
        :param is_locked: 是否触发紧急加锁
        :param rec_seconds: 当前分段录像时长 (秒)
        :param temp_c: SoC 芯片温度 (°C)
        :param total_g: 当前 IMU 总加速度 (G)
        :param snap_count: 累计车辆/事件抓拍张数
        :param yaw_rate_dps: IMU 实时偏航角速度 (deg/s)，驱动动态弯道地毯随动
        :param road_res: 实际道路感知结果 (RoadDetectionResult)
        :param horizon_y: 自适应动态地平线纵坐标 (像素)
        :param acc_y: IMU 实时 Y 轴重力加速度 (m/s²)
        :param gravity_direction: 重力朝向 (DOWN / UP)
        :param is_inverted: 相机是否处于倒装/吊装姿态 (True 时 HUD 整体模块自动旋转适配物理屏幕)
        """
        # 1. 绘制前向主车道 ROI 参考引导框 (轻量淡蓝色角标)
        roi_x = int(self.config.IMG_WIDTH * self.config.ROI_X_MIN_RATIO)
        roi_w = int(self.config.IMG_WIDTH * (self.config.ROI_X_MAX_RATIO - self.config.ROI_X_MIN_RATIO))
        roi_y = int(self.config.IMG_HEIGHT * self.config.ROI_Y_MIN_RATIO)
        roi_h = int(self.config.IMG_HEIGHT * (1.0 - self.config.ROI_Y_MIN_RATIO))
        img.draw_rect(roi_x, roi_y, roi_w, roi_h, color=self.COLOR_CYAN, thickness=1)

        # 2. 绘制实际物理道路左右边界 (Real Road Perception)
        if road_res is not None and getattr(road_res, "detected", False):
            self.render_actual_road(img, road_res)

        # 3. 绘制虚拟车道透视安全地毯 (Lane Corridor Envelope, 支持动态弯道随动)
        self.render_lane_corridor(img, alerts, yaw_rate_dps=yaw_rate_dps)

        # 4. 绘制每个被跟踪目标与距离/TTC 徽标
        for t_id, target in tracks.items():
            x, y, w, h = target.bbox
            cls_name = self.CLASS_NAMES.get(target.class_id, "Obj")
            dist_str = f"{target.distance_m:.1f}m"

            # 判断该目标经过虚拟车道走廊与防抖后的生效告警级别
            active_lvl = getattr(target, "active_alert_level", AlertLevel.NONE)
            in_corridor = getattr(target, "in_lane_corridor", True)
            corridor_tag = "" if in_corridor else " [Neighbor]"

            if active_lvl == AlertLevel.CRITICAL:
                box_color = self.COLOR_CRITICAL
                badge_text = f"[#{t_id}] {cls_name} {dist_str} | TTC:{target.ttc_sec:.1f}s BRAKE!"
                thickness = 3
            elif active_lvl == AlertLevel.WARNING:
                box_color = self.COLOR_WARNING
                badge_text = f"[#{t_id}] {cls_name} {dist_str} | TTC:{target.ttc_sec:.1f}s"
                thickness = 2
            else:
                box_color = self.COLOR_SAFE
                badge_text = f"[#{t_id}] {cls_name} {dist_str}{corridor_tag}"
                thickness = 1

            img.draw_rect(int(x), int(y), int(w), int(h), color=box_color, thickness=thickness)
            label_y = max(10, int(y) - 18)
            img.draw_string(int(x), label_y, badge_text, color=box_color, scale=1.0)

        # 5. 顶部全局紧急碰撞警报横幅 (如果有 CRITICAL 告警)
        has_critical = any(a["level"] == AlertLevel.CRITICAL for a in alerts)
        if has_critical:
            banner_y = 30 if not is_inverted else self.config.IMG_HEIGHT - 66
            def draw_critical_content(target, ox, oy):
                target.draw_string(ox + 20, oy + 8, "CRITICAL: COLLISION ALERT! BRAKE!", color=self.COLOR_WHITE, scale=1.3)
            self.draw_rotated_bar(img, 400, 36, 120, banner_y, self.COLOR_CRITICAL, draw_critical_content, is_inverted)

        # 6. 状态顶栏 (录像状态、加锁状态、中央版本徽章、温度与 G 值)
        # 在正装模式下位于图像 y=0~32；在倒装模式下位于图像 y=H-32~H (对应肉眼物理屏幕上方)
        status_bar_y = 0 if not is_inverted else self.config.IMG_HEIGHT - 32
        rec_time_str = self.format_duration(rec_seconds)
        if is_locked:
            rec_status = f"REC [LOCKED] ({rec_time_str})"
            rec_color = self.COLOR_WARNING
        elif is_recording:
            rec_status = f"REC [LIVE] ({rec_time_str})"
            rec_color = self.COLOR_CRITICAL
        else:
            rec_status = "REC [PAUSED]"
            rec_color = self.COLOR_WHITE

        ver_text = f"Camp Dashcam {self.config.APP_VERSION}" if not has_critical else ""
        status_right = f"G:{total_g:.2f}G | SoC:{temp_c}C"

        def draw_status_content(target, ox, oy):
            target.draw_string(ox + 10, oy + 8, rec_status, color=rec_color, scale=1.0)
            if ver_text:
                target.draw_string(ox + 240, oy + 8, ver_text, color=self.COLOR_CYAN, scale=1.0)
            target.draw_string(ox + self.config.IMG_WIDTH - 190, oy + 8, status_right, color=self.COLOR_WHITE, scale=1.0)

        self.draw_rotated_bar(img, self.config.IMG_WIDTH, 32, 0, status_bar_y, self.COLOR_DARK_BG, draw_status_content, is_inverted)

        # 7. 底部信息条 (抓拍统计与退出触摸区域)
        # 在正装模式下位于图像 y=H-26~H；在倒装模式下位于图像 y=0~26 (对应肉眼物理屏幕下方)
        info_bar_y = self.config.IMG_HEIGHT - 26 if not is_inverted else 0

        def draw_info_content(target, ox, oy):
            target.draw_string(ox + 10, oy + 6, f"Snaps: {snap_count}  |  ROI: Main Lane", color=self.COLOR_WHITE, scale=0.9)
            target.draw_string(ox + self.config.IMG_WIDTH - 85, oy + 6, "[ Exit ]", color=self.COLOR_WARNING, scale=0.9)

        self.draw_rotated_bar(img, self.config.IMG_WIDTH, 26, 0, info_bar_y, self.COLOR_DARK_BG, draw_info_content, is_inverted)


