"""
Unit tests for Telemetry Logger & Vehicle Snapshot Manager
"""

import unittest
import os
import shutil
import tempfile
import time
from unittest.mock import MagicMock
from core.config import DashcamConfig
from core.fcw_tracker import FCWTarget, AlertLevel
from core.telemetry_logger import TelemetryLogger, SnapshotManager


class TestTelemetryAndSnapshot(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="telemetry_test_")
        self.config = DashcamConfig
        self.logger = TelemetryLogger(log_dir=self.test_dir, config=self.config)
        self.snapshot_mgr = SnapshotManager(snapshots_dir=self.test_dir, config=self.config)

    def tearDown(self):
        if self.logger:
            self.logger.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)


    def test_telemetry_csv_headers_and_row_format(self):
        """
        验证遥测 CSV 文件在初始化时正确写入表头，并能连续写入结构化遥测数据。
        """
        self.assertTrue(os.path.exists(self.logger.current_csv_path))

        # 写入一条遥测行
        self.logger.log_frame(
            timestamp=1700000000.0,
            fps=29.5,
            temp_c=48,
            total_g=1.02,
            dyn_g=0.02,
            imu_raw=[0.5, 0.5, 9.8, 0.0, 0.0, 0.0],
            target_count=2,
            lead_id=1,
            lead_dist=18.5,
            rel_speed=2.0,
            ttc=9.25,
            alert_level=AlertLevel.NONE,
            is_locked=False
        )

        self.logger.flush()

        with open(self.logger.current_csv_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        self.assertGreaterEqual(len(lines), 2)
        # 表头检查
        self.assertIn("timestamp", lines[0])
        self.assertIn("lead_dist", lines[0])
        self.assertIn("ttc", lines[0])
        # 数据行检查
        self.assertIn("1700000000.0", lines[1])
        self.assertIn("18.5", lines[1])
        self.assertIn("NONE", lines[1])

    def test_snapshot_trigger_on_new_vehicle_and_cooldown(self):
        """
        验证车辆识别抓拍与冷却机制：
        1. 首次在主车道识别到车辆时，触发一次抓拍；
        2. 在冷却时间内（如 5 秒），同一 Track ID 的车辆不重复抓拍；
        3. 当该车辆触发 CRITICAL 紧急逼近时，立即触发紧急事件抓拍。
        """
        mock_img = MagicMock()
        target = FCWTarget(track_id=10, class_id=DashcamConfig.TARGET_CAR, bbox=[200, 200, 100, 100], distance_m=20.0, timestamp=100.0)

        # 1. 首次识别到车辆 -> 应当触发抓拍
        should_snap1 = self.snapshot_mgr.should_take_snapshot(target, alert_level=AlertLevel.NONE, now_sec=100.0)
        self.assertTrue(should_snap1)

        saved_path1 = self.snapshot_mgr.save_snapshot(mock_img, target, alert_level=AlertLevel.NONE, now_sec=100.0)
        self.assertIsNotNone(saved_path1)
        self.assertTrue(mock_img.save.called)

        # 2. 2 秒后（冷却期内）正常跟车 -> 不应重复抓拍
        should_snap2 = self.snapshot_mgr.should_take_snapshot(target, alert_level=AlertLevel.NONE, now_sec=102.0)
        self.assertFalse(should_snap2)

        # 3. 3 秒后（处于紧急碰撞告警） -> 应当立刻触发紧急事件抓拍
        target.ttc_sec = 1.4
        should_snap3 = self.snapshot_mgr.should_take_snapshot(target, alert_level=AlertLevel.CRITICAL, now_sec=103.0)
        self.assertTrue(should_snap3, "进入 CRITICAL 紧急预警时应突破常规跟车冷却，立即抓拍！")

    def test_touchscreen_1d_coordinate_parsing(self):
        """
        验证 MaixPy 1D 触控输入格式 [x, y, pressed] 正确解析与退出判据。
        """
        from apps.camp_dashcam.main import parse_touch_exit

        # 正常点击左下角退出区域 (x=50, y=460, pressed=1)
        touch_exit = [50, 460, 1]
        self.assertTrue(parse_touch_exit(touch_exit, img_w=640, img_h=480))

        # 点击画面中间非退出区域 (x=320, y=240, pressed=1)
        touch_center = [320, 240, 1]
        self.assertFalse(parse_touch_exit(touch_center, img_w=640, img_h=480))

        # 未按下状态 (pressed=0)
        touch_unpressed = [50, 460, 0]
        self.assertFalse(parse_touch_exit(touch_unpressed, img_w=640, img_h=480))

        # 空数据/None 输入
        self.assertFalse(parse_touch_exit(None, img_w=640, img_h=480))
        self.assertFalse(parse_touch_exit([], img_w=640, img_h=480))


if __name__ == "__main__":
    unittest.main()
