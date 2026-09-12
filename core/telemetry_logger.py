"""
Telemetry Logger & Vehicle Snapshot Manager for MaixCAM Dashcam
"""

import os
import time
try:
    from config import DashcamConfig
    from fcw_tracker import AlertLevel, FCWTarget
except ImportError:
    from .config import DashcamConfig
    from .fcw_tracker import AlertLevel, FCWTarget


class TelemetryLogger:
    CSV_HEADER = "timestamp,fps,soc_temp,total_g,dyn_g,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,target_count,lead_id,lead_dist,rel_speed,ttc,alert_level,is_locked\n"

    def __init__(self, log_dir: str = None, config=DashcamConfig):
        self.config = config
        self.log_dir = log_dir or os.path.join(config.DEFAULT_STORAGE_BASE, "telemetry")
        os.makedirs(self.log_dir, exist_ok=True)

        # 启动时创建当前行程专用的 CSV 文件
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.current_csv_path = os.path.join(self.log_dir, f"telemetry_{time_str}.csv")
        self.file_handle = None
        self._init_csv()

    def _init_csv(self):
        try:
            self.file_handle = open(self.current_csv_path, "a", encoding="utf-8")
            if os.path.getsize(self.current_csv_path) == 0:
                self.file_handle.write(self.CSV_HEADER)
                self.file_handle.flush()
        except Exception as e:
            print(f"⚠️ [TelemetryLogger] 创建遥测文件失败: {e}")

    def log_frame(self, timestamp: float, fps: float, temp_c: int, total_g: float, dyn_g: float,
                  imu_raw: list, target_count: int, lead_id: int, lead_dist: float, rel_speed: float,
                  ttc: float, alert_level: str, is_locked: bool):
        """记录一帧结构化遥测数据 (包含 6 轴 IMU 原始动力学数据)"""
        if not self.file_handle:
            return

        ttc_str = f"{ttc:.2f}" if ttc != float("inf") else "inf"
        dist_str = f"{lead_dist:.2f}" if lead_dist > 0 else "0.00"
        speed_str = f"{rel_speed:.2f}"

        # 6轴惯性原始数据 (ax, ay, az, gx, gy, gz)
        ax = imu_raw[0] if imu_raw and len(imu_raw) > 0 else 0.0
        ay = imu_raw[1] if imu_raw and len(imu_raw) > 1 else 0.0
        az = imu_raw[2] if imu_raw and len(imu_raw) > 2 else 0.0
        gx = imu_raw[3] if imu_raw and len(imu_raw) > 3 else 0.0
        gy = imu_raw[4] if imu_raw and len(imu_raw) > 4 else 0.0
        gz = imu_raw[5] if imu_raw and len(imu_raw) > 5 else 0.0

        line = (f"{timestamp:.3f},{fps:.1f},{temp_c},{total_g:.2f},{dyn_g:.2f},"
                f"{ax:.2f},{ay:.2f},{az:.2f},{gx:.2f},{gy:.2f},{gz:.2f},"
                f"{target_count},{lead_id},{dist_str},{speed_str},{ttc_str},"
                f"{alert_level},{1 if is_locked else 0}\n")
        try:
            self.file_handle.write(line)
        except Exception:
            pass

    def flush(self):

        if self.file_handle:
            try:
                self.file_handle.flush()
            except Exception:
                pass

    def close(self):
        if self.file_handle:
            try:
                self.file_handle.close()
            except Exception:
                pass
            self.file_handle = None


class SnapshotManager:
    CLASS_NAMES = {
        DashcamConfig.TARGET_PERSON: "person",
        DashcamConfig.TARGET_BICYCLE: "bike",
        DashcamConfig.TARGET_CAR: "car",
        DashcamConfig.TARGET_MOTORCYCLE: "motor",
        DashcamConfig.TARGET_BUS: "bus",
        DashcamConfig.TARGET_TRUCK: "truck"
    }

    def __init__(self, snapshots_dir: str = None, config=DashcamConfig, cooldown_sec: float = 5.0):
        self.config = config
        self.snapshots_dir = snapshots_dir or os.path.join(config.DEFAULT_STORAGE_BASE, "snapshots")
        self.cooldown_sec = cooldown_sec
        self.track_history = {}  # {track_id: {'last_time': float, 'last_level': str}}
        self.snapshot_count = 0
        os.makedirs(self.snapshots_dir, exist_ok=True)

    def should_take_snapshot(self, target: FCWTarget, alert_level: str, now_sec: float) -> bool:
        """
        判断是否需要对该目标进行自动抓拍：
        1. 首次识别到该目标时抓拍；
        2. 目标恶化进入 CRITICAL / WARNING 预警时抓拍；
        3. 常态下同一目标限制在 cooldown_sec 冷却期内不重复堆积。
        """
        t_id = target.track_id
        if t_id not in self.track_history:
            return True

        hist = self.track_history[t_id]
        last_time = hist.get("last_time", 0.0)
        last_level = hist.get("last_level", AlertLevel.NONE)

        # 危险升级事件突破常规冷却
        if alert_level == AlertLevel.CRITICAL and last_level != AlertLevel.CRITICAL:
            return True
        if alert_level == AlertLevel.WARNING and last_level == AlertLevel.NONE and (now_sec - last_time > 2.0):
            return True

        # 常规时间周期性抓拍 (例如 10 秒以上)
        if now_sec - last_time >= (self.cooldown_sec * 2):
            return True

        return False

    def save_snapshot(self, img, target: FCWTarget, alert_level: str, now_sec: float) -> str:
        """保存单帧抓拍图片"""
        t_id = target.track_id
        cls_name = self.CLASS_NAMES.get(target.class_id, "obj")
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(now_sec))
        filename = f"snap_{time_str}_id{t_id}_{cls_name}_{alert_level}.jpg"
        filepath = os.path.join(self.snapshots_dir, filename)

        try:
            img.save(filepath)
            self.snapshot_count += 1
            self.track_history[t_id] = {
                "last_time": now_sec,
                "last_level": alert_level
            }
            print(f"📸 [Snapshot] 自动保存高价值抓拍: {filename} (距:{target.distance_m:.1f}m, TTC:{target.ttc_sec:.1f}s)")
            return filepath
        except Exception as e:
            print(f"⚠️ [Snapshot] 保存快照失败: {e}")
            return None
