"""
Unit tests for PipelineManager & SharedADASState (Decoupled 4-Pipeline Orchestration)
"""

import unittest
import os
import shutil
import tempfile
import time
import numpy as np
from unittest.mock import MagicMock, patch

from core.config import DashcamConfig
from core.pipeline_manager import PipelineManager, SharedADASState, PerceptionSnapshot


class TestPipelineManager(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="dashcam_test_pipeline_")
        self.config = DashcamConfig
        self.config.DEFAULT_STORAGE_BASE = self.test_dir

        self.mock_cam = MagicMock()
        self.mock_detector = MagicMock()
        self.mock_imu = MagicMock()
        self.mock_disp = MagicMock()

        self.pipeline = PipelineManager(
            config=self.config,
            camera_dev=self.mock_cam,
            detector_dev=self.mock_detector,
            imu_dev=self.mock_imu,
            display_dev=self.mock_disp
        )

    def tearDown(self):
        if self.pipeline._running:
            self.pipeline.stop(timeout=1.0)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_shared_adas_state_thread_safety(self):
        """
        验证 SharedADASState 更新与快照读取原子性
        """
        state = SharedADASState()
        state.update_kinematics(
            tracks={}, alerts=[], is_locked=True, is_inverted=True,
            gravity_direction="UP", smooth_ay=-9.8, horizon_y=240,
            yaw_rate_dps=1.5, total_g=1.05, dyn_g=0.05
        )
        snap = state.get_snapshot()
        self.assertTrue(snap["is_locked"])
        self.assertTrue(snap["is_inverted"])
        self.assertEqual(snap["gravity_direction"], "UP")
        self.assertEqual(snap["yaw_rate_dps"], 1.5)

        state.update_system_stats(rec_seconds=120, temp_c=52, snap_count=3, fps=29.5)
        snap2 = state.get_snapshot()
        self.assertEqual(snap2["rec_seconds"], 120)
        self.assertEqual(snap2["temp_c"], 52)
        self.assertEqual(snap2["snap_count"], 3)
        self.assertEqual(snap2["fps"], 29.5)

    def test_pipeline_manager_step_single_cycle(self):
        """
        验证 PipelineManager 单步无硬件 Mock 执行
        """
        dummy_img = MagicMock()
        dummy_imu = [0.0, 9.8, 0.0, 0.0, 0.0, 0.0]

        snapshot = self.pipeline.step_single_cycle(mock_img=dummy_img, mock_imu=dummy_imu, now_sec=100.0)
        self.assertIn("tracks", snapshot)
        self.assertIn("alerts", snapshot)
        self.assertFalse(snapshot["is_inverted"])
        self.assertEqual(snapshot["gravity_direction"], "DOWN")

    def test_pipeline_manager_multithread_lifecycle(self):
        """
        验证真实多线程启动与退出
        """
        # 设置 mock 传感器返回
        self.mock_cam.read.return_value = MagicMock()
        self.mock_imu.read.return_value = [0.0, 9.8, 0.0, 0.0, 0.0, 0.0]
        self.mock_detector.detect.return_value = []

        self.pipeline.start()
        self.assertTrue(self.pipeline._running)
        time.sleep(0.15)  # 允许各流水线线程跑几轮

        self.pipeline.stop(timeout=1.0)
        self.assertFalse(self.pipeline._running)

    def test_kinematics_shock_and_critical_lock_trigger(self):
        """
        验证高加速度冲击 (shock) 与 CRITICAL 告警自动触发录像加锁
        """
        # 模拟剧烈碰撞 IMU (30 m/s^2)
        self.mock_cam.read.return_value = MagicMock()
        self.mock_imu.read.return_value = [0.0, 30.0, 0.0, 0.0, 0.0, 0.0]

        # 模拟前车极度逼近以产生 CRITICAL 告警
        mock_obj = MagicMock()
        mock_obj.x, mock_obj.y, mock_obj.w, mock_obj.h = 280, 420, 80, 80
        mock_obj.track_id = 1
        mock_obj.class_id = DashcamConfig.TARGET_CAR
        self.mock_detector.detect.return_value = [mock_obj]

        self.pipeline.start()
        time.sleep(0.15)

        self.pipeline.stop(timeout=1.0)
        snap = self.pipeline.shared_state.get_snapshot()
        self.assertTrue(snap["is_locked"], "检测到剧烈碰撞时必须自动切换为锁定状态！")

    def test_pipeline_road_detection_and_safe_exceptions(self):
        """
        验证道路感知分支触发、重复 start/stop 以及外设异常容错
        """
        # 1. 重复 start / stop
        self.pipeline.start()
        self.pipeline.start()
        self.pipeline.stop(timeout=1.0)
        self.pipeline.stop(timeout=1.0)

        # 2. 模拟包含 bgr 属性的图像以触发道路检测分支
        mock_img_with_bgr = MagicMock()
        mock_img_with_bgr.bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        self.mock_cam.read.return_value = mock_img_with_bgr
        self.mock_disp.show.side_effect = RuntimeError("Display bus busy")

        self.pipeline.start()
        time.sleep(0.15)
        self.pipeline.stop(timeout=1.0)


if __name__ == '__main__':
    unittest.main()
