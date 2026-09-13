"""
Pipeline Manager & Decoupled Workers for MaixCAM Dashcam (Issue #2)
Orchestrates Perception, Kinematics/FCW, Display, and Async Storage pipelines.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple

try:
    from config import DashcamConfig
    from fcw_tracker import FCWAnalyzer, FCWTarget, AlertLevel, OrientationDetector, GravityDirection
    from g_sensor_guard import GSensorGuard
    from road_detector import RoadDetector, RoadDetectionResult
    from hud_renderer import HUDRenderer
    from telemetry_logger import TelemetryLogger, SnapshotManager
    from storage_manager import StorageManager
    from video_recorder import VideoRecorder
    from async_storage_worker import AsyncStorageWorker, TelemetryPacket
except ImportError:
    from .config import DashcamConfig
    from .fcw_tracker import FCWAnalyzer, FCWTarget, AlertLevel, OrientationDetector, GravityDirection
    from .g_sensor_guard import GSensorGuard
    from .road_detector import RoadDetector, RoadDetectionResult
    from .hud_renderer import HUDRenderer
    from .telemetry_logger import TelemetryLogger, SnapshotManager
    from .storage_manager import StorageManager
    from .video_recorder import VideoRecorder
    from .async_storage_worker import AsyncStorageWorker, TelemetryPacket


@dataclass
class PerceptionSnapshot:
    """不可变感知快照 (由感知线程产出，供动力学/FCW 线程消费)"""
    timestamp: float = 0.0
    detections: List[Dict[str, Any]] = field(default_factory=list)
    road_result: Optional[RoadDetectionResult] = None


class SharedADASState:
    """
    线程安全的 ADAS 聚合状态池：
    - 生产者：Kinematics & FCW 线程 (毫秒级写入最新 tracks、alerts、attitude)
    - 消费者：Display 线程 (每帧无锁/快速快照读取，驱动 HUD 渲染并送显)
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.tracks: Dict[int, FCWTarget] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.is_locked: bool = False
        self.is_inverted: bool = False
        self.gravity_direction: str = GravityDirection.DOWN
        self.smooth_ay: float = 9.8
        self.horizon_y: int = 240
        self.yaw_rate_dps: float = 0.0
        self.total_g: float = 1.0
        self.dyn_g: float = 0.0
        self.road_res: Optional[RoadDetectionResult] = None
        self.rec_seconds: int = 0
        self.temp_c: int = 45
        self.snap_count: int = 0
        self.fps: float = 30.0

    def update_kinematics(
        self,
        tracks: Dict[int, FCWTarget],
        alerts: List[Dict[str, Any]],
        is_locked: bool,
        is_inverted: bool,
        gravity_direction: str,
        smooth_ay: float,
        horizon_y: int,
        yaw_rate_dps: float,
        total_g: float,
        dyn_g: float,
        road_res: Optional[RoadDetectionResult] = None
    ):
        """动力学线程原子更新当前 ADAS 状态"""
        with self._lock:
            self.tracks = dict(tracks)
            self.alerts = list(alerts)
            self.is_locked = is_locked
            self.is_inverted = is_inverted
            self.gravity_direction = gravity_direction
            self.smooth_ay = smooth_ay
            self.horizon_y = horizon_y
            self.yaw_rate_dps = yaw_rate_dps
            self.total_g = total_g
            self.dyn_g = dyn_g
            if road_res is not None:
                self.road_res = road_res

    def update_system_stats(self, rec_seconds: int, temp_c: int, snap_count: int, fps: float):
        """更新系统仪表状态"""
        with self._lock:
            self.rec_seconds = rec_seconds
            self.temp_c = temp_c
            self.snap_count = snap_count
            self.fps = fps

    def get_snapshot(self) -> Dict[str, Any]:
        """获取一份极轻量的渲染状态快照 (线程安全，拷贝仅耗费数微秒)"""
        with self._lock:
            return {
                "tracks": self.tracks,
                "alerts": self.alerts,
                "is_locked": self.is_locked,
                "is_inverted": self.is_inverted,
                "gravity_direction": self.gravity_direction,
                "smooth_ay": self.smooth_ay,
                "horizon_y": self.horizon_y,
                "yaw_rate_dps": self.yaw_rate_dps,
                "total_g": self.total_g,
                "dyn_g": self.dyn_g,
                "road_res": self.road_res,
                "rec_seconds": self.rec_seconds,
                "temp_c": self.temp_c,
                "snap_count": self.snap_count,
                "fps": self.fps
            }


class PipelineManager:
    """
    四流水线中枢调度器：
    1. Pipeline 1 (Perception): 摄像头采集 + YOLO11 NPU 目标检测 + 路面边缘
    2. Pipeline 2 (Kinematics & FCW): 50Hz IMU 姿态解算 + FCW 追踪与 TTC 矩阵
    3. Pipeline 3 (Display & UI): 30 FPS 渲染画面 + 矢量 HUD + 硬件直显
    4. Pipeline 4 (Async Storage): 异步分段录像写盘 + 遥测追加 + 物理刷盘
    """

    def __init__(
        self,
        config=DashcamConfig,
        camera_dev=None,
        detector_dev=None,
        imu_dev=None,
        display_dev=None,
        touch_dev=None
    ):
        self.config = config
        self.camera_dev = camera_dev
        self.detector_dev = detector_dev
        self.imu_dev = imu_dev
        self.display_dev = display_dev
        self.touch_dev = touch_dev

        # 核心算法与数据组件
        self.fcw_analyzer = FCWAnalyzer(config=config)
        self.orientation_detector = OrientationDetector(config=config)
        self.g_guard = GSensorGuard(config=config)
        self.road_detector = RoadDetector(config=config)
        self.hud_renderer = HUDRenderer(config=config)
        self.storage_mgr = StorageManager(config=config)
        self.telemetry_logger = TelemetryLogger(config=config)
        self.snapshot_mgr = SnapshotManager(config=config)
        self.video_recorder = VideoRecorder(storage_manager=self.storage_mgr, config=config)

        # 异步存储工作流水线
        self.storage_worker = AsyncStorageWorker(
            storage_manager=self.storage_mgr,
            video_recorder=self.video_recorder,
            telemetry_logger=self.telemetry_logger,
            config=config
        )

        # 共享状态与数据通道
        self.shared_state = SharedADASState()
        self._latest_perception = PerceptionSnapshot()
        self._perception_lock = threading.Lock()
        self._latest_frame = None
        self._frame_lock = threading.Lock()

        self._running = False
        self._threads: List[threading.Thread] = []

        # 运行统计
        self.start_time = 0.0
        self.frame_count = 0
        self.fps = 30.0

    def start(self):
        """启动全部流水线线程"""
        if self._running:
            return
        self._running = True
        self.start_time = time.time()

        # 1. 启动异步存储流水线
        self.storage_worker.start()

        # 2. 启动感知流水线 (Pipeline 1)
        t_perception = threading.Thread(target=self._perception_loop, name="PerceptionPipe", daemon=True)
        # 3. 启动动力学与 FCW 流水线 (Pipeline 2)
        t_kinematics = threading.Thread(target=self._kinematics_loop, name="KinematicsPipe", daemon=True)
        # 4. 启动显示与 UI 渲染流水线 (Pipeline 3)
        t_display = threading.Thread(target=self._display_loop, name="DisplayPipe", daemon=True)

        self._threads = [t_perception, t_kinematics, t_display]
        for t in self._threads:
            t.start()

    def stop(self, timeout: float = 2.0):
        """安全平稳停机并释放硬件句柄"""
        if not self._running:
            return
        self._running = False
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=timeout)

        self.storage_worker.stop(timeout=timeout)

    def _perception_loop(self):
        """Pipeline 1: 图像捕获与视觉感知处理"""
        while self._running:
            img = None
            if self.camera_dev:
                try:
                    img = self.camera_dev.read()
                except Exception:
                    img = None

            if img is None:
                time.sleep(0.01)
                continue

            # 保存最新图像帧供 UI 和录像使用
            with self._frame_lock:
                self._latest_frame = img

            # 提交一帧到异步写盘队列
            self.storage_worker.submit_frame(img)

            # NPU YOLO 检测
            detections = []
            if self.detector_dev:
                try:
                    objs = self.detector_dev.detect(img, conf_th=0.35, iou_th=0.45)
                    for obj in objs:
                        detections.append({
                            "track_id": getattr(obj, "track_id", getattr(obj, "id", 1)),
                            "class_id": getattr(obj, "class_id", 0),
                            "bbox": [obj.x, obj.y, obj.w, obj.h]
                        })
                except Exception:
                    pass

            # 道路标线感知
            road_res = None
            if getattr(self.config, "ROAD_DETECTION_ENABLED", True):
                try:
                    cv_img = getattr(img, "bgr", None)
                    if cv_img is not None:
                        snap = self.shared_state.get_snapshot()
                        road_res = self.road_detector.detect_actual_road(
                            cv_img,
                            horizon_y=int(snap["horizon_y"]),
                            imu_yaw_rate=snap["yaw_rate_dps"],
                            is_inverted=snap["is_inverted"]
                        )
                except Exception:
                    pass

            # 原子更新感知快照
            with self._perception_lock:
                self._latest_perception = PerceptionSnapshot(
                    timestamp=time.time(),
                    detections=detections,
                    road_result=road_res
                )

    def _kinematics_loop(self):
        """Pipeline 2: 50Hz IMU 动力学、姿态自校准与 FCW 评估"""
        interval = 0.02  # 50Hz
        while self._running:
            loop_start = time.time()
            now_sec = loop_start

            imu_data = None
            if self.imu_dev:
                try:
                    imu_data = self.imu_dev.read()
                except Exception:
                    imu_data = None

            total_g = 1.0
            dyn_g = 0.0
            yaw_rate_dps = 0.0

            if imu_data and len(imu_data) >= 3:
                acc_x, acc_y, acc_z = imu_data[0], imu_data[1], imu_data[2]
                total_g = (acc_x**2 + acc_y**2 + acc_z**2)**0.5 / 9.80665
                dyn_g = abs(total_g - 1.0)

                # 更新朝向
                self.orientation_detector.update(acc_x, acc_y, acc_z)
                inv = self.orientation_detector.is_inverted
                acc_x_eff = -acc_x if inv else acc_x
                acc_y_eff = -acc_y if inv else acc_y
                self.fcw_analyzer.update_pitch_with_gravity(acc_x_eff, acc_y_eff, acc_z)

                # 评估冲击
                shock = self.g_guard.evaluate_imu(imu_data, now_sec=now_sec)
                if shock:
                    self.g_guard.is_locked = True
                    self.storage_worker.request_lock_current_segment()

            if imu_data and len(imu_data) >= 5:
                yaw_rate_dps = float(imu_data[4])

            # 获取最新感知数据
            with self._perception_lock:
                perc = self._latest_perception

            is_inv = self.orientation_detector.is_inverted
            alerts = self.fcw_analyzer.process_detections(
                perc.detections, now_sec=now_sec, yaw_rate_dps=yaw_rate_dps, is_inverted=is_inv
            )

            # CRITICAL 预警触发录像加锁
            if any(a["level"] == AlertLevel.CRITICAL for a in alerts):
                self.storage_worker.request_lock_current_segment()

            # 结构化遥测数据推送
            lead_t = min(self.fcw_analyzer.tracks.values(), key=lambda t: t.distance_m, default=None)
            lead_id = lead_t.track_id if lead_t else -1
            lead_dist = lead_t.distance_m if lead_t else 0.0
            lead_speed = lead_t.rel_speed_mps if lead_t else 0.0
            lead_ttc = lead_t.ttc_sec if lead_t else float("inf")
            highest_alert = alerts[0]["level"] if alerts else AlertLevel.NONE

            telem = TelemetryPacket(
                timestamp=now_sec,
                fps=self.fps,
                temp_c=45,
                total_g=total_g,
                dyn_g=dyn_g,
                imu_raw=imu_data,
                target_count=len(self.fcw_analyzer.tracks),
                lead_id=lead_id,
                lead_dist=lead_dist,
                rel_speed=lead_speed,
                ttc=lead_ttc,
                alert_level=highest_alert,
                is_locked=self.g_guard.is_locked
            )
            self.storage_worker.submit_telemetry(telem)

            # 发布 ADAS 状态给 UI 线程
            self.shared_state.update_kinematics(
                tracks=self.fcw_analyzer.tracks,
                alerts=alerts,
                is_locked=self.g_guard.is_locked,
                is_inverted=is_inv,
                gravity_direction=self.orientation_detector.gravity_direction,
                smooth_ay=self.orientation_detector.smooth_ay,
                horizon_y=int(self.fcw_analyzer.current_horizon_y),
                yaw_rate_dps=yaw_rate_dps,
                total_g=total_g,
                dyn_g=dyn_g,
                road_res=perc.road_result
            )

            elapsed = time.time() - loop_start
            sleep_time = max(0.002, interval - elapsed)
            time.sleep(sleep_time)

    def _display_loop(self):
        """Pipeline 3: 30 FPS UI 渲染与屏幕呈现 (永不阻塞)"""
        target_fps = 30.0
        frame_interval = 1.0 / target_fps
        last_calc_time = time.time()
        fps_counter = 0

        while self._running:
            t0 = time.time()
            fps_counter += 1
            if t0 - last_calc_time >= 1.0:
                self.fps = fps_counter / (t0 - last_calc_time)
                fps_counter = 0
                last_calc_time = t0

            # 获取当前帧
            img = None
            with self._frame_lock:
                img = self._latest_frame

            if img is not None:
                # 获取轻量快照
                state = self.shared_state.get_snapshot()
                rec_duration = int(t0 - self.start_time)

                # 渲染 HUD
                self.hud_renderer.render_hud(
                    img=img,
                    tracks=state["tracks"],
                    alerts=state["alerts"],
                    is_recording=True,
                    is_locked=state["is_locked"],
                    rec_seconds=rec_duration,
                    temp_c=state["temp_c"],
                    total_g=state["total_g"],
                    snap_count=self.snapshot_mgr.snapshot_count,
                    yaw_rate_dps=state["yaw_rate_dps"],
                    road_res=state["road_res"],
                    horizon_y=state["horizon_y"],
                    acc_y=state["smooth_ay"],
                    gravity_direction=state["gravity_direction"],
                    is_inverted=state["is_inverted"]
                )

                if self.display_dev:
                    try:
                        self.display_dev.show(img)
                    except Exception:
                        pass

            elapsed = time.time() - t0
            sleep_sec = max(0.005, frame_interval - elapsed)
            time.sleep(sleep_sec)

    def step_single_cycle(self, mock_img=None, mock_imu=None, now_sec: float = None):
        """
        单步执行接口：专为无硬件单元测试与 CI 设计，避免在测试中开启多线程。
        """
        ts = now_sec if now_sec is not None else time.time()
        if mock_img is not None:
            with self._frame_lock:
                self._latest_frame = mock_img
            self.storage_worker.submit_frame(mock_img, timestamp=ts)

        # 动力学与 FCW 评估
        if mock_imu and len(mock_imu) >= 3:
            self.orientation_detector.update(mock_imu[0], mock_imu[1], mock_imu[2])
            inv = self.orientation_detector.is_inverted
            acc_x_eff = -mock_imu[0] if inv else mock_imu[0]
            acc_y_eff = -mock_imu[1] if inv else mock_imu[1]
            self.fcw_analyzer.update_pitch_with_gravity(acc_x_eff, acc_y_eff, mock_imu[2])

        is_inv = self.orientation_detector.is_inverted
        alerts = self.fcw_analyzer.process_detections([], now_sec=ts, is_inverted=is_inv)

        self.shared_state.update_kinematics(
            tracks=self.fcw_analyzer.tracks,
            alerts=alerts,
            is_locked=self.g_guard.is_locked,
            is_inverted=is_inv,
            gravity_direction=self.orientation_detector.gravity_direction,
            smooth_ay=self.orientation_detector.smooth_ay,
            horizon_y=int(self.fcw_analyzer.current_horizon_y),
            yaw_rate_dps=0.0,
            total_g=1.0,
            dyn_g=0.0
        )
        return self.shared_state.get_snapshot()
