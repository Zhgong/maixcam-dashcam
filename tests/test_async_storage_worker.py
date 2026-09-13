"""
Unit tests for AsyncStorageWorker (Drop-oldest Queue & Background Flush)
"""

import unittest
import os
import shutil
import tempfile
import time
import numpy as np
from unittest.mock import MagicMock, patch

from core.config import DashcamConfig
from core.storage_manager import StorageManager
from core.video_recorder import VideoRecorder
from core.telemetry_logger import TelemetryLogger
from core.async_storage_worker import AsyncStorageWorker, RecordFramePacket, TelemetryPacket


class TestAsyncStorageWorker(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="dashcam_test_async_")
        self.config = DashcamConfig
        self.storage_mgr = StorageManager(base_dir=self.test_dir, config=self.config)
        self.telemetry = TelemetryLogger(log_dir=os.path.join(self.test_dir, "telemetry"), config=self.config)
        self.recorder = VideoRecorder(storage_manager=self.storage_mgr, config=self.config)

        self.worker = AsyncStorageWorker(
            storage_manager=self.storage_mgr,
            video_recorder=self.recorder,
            telemetry_logger=self.telemetry,
            config=self.config,
            queue_maxsize=5
        )

    def tearDown(self):
        if self.worker._running:
            self.worker.stop(timeout=1.0)
        self.telemetry.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_worker_lifecycle_start_and_stop(self):
        """
        验证后台 Worker 正常启动、退出与重复 start/stop 安全性。
        """
        self.assertFalse(self.worker._running)
        self.worker.start()
        self.assertTrue(self.worker._running)
        # 重复 start 无副作用
        self.worker.start()
        self.assertTrue(self.worker._running)

        self.worker.stop(timeout=1.0)
        self.assertFalse(self.worker._running)
        # 重复 stop 无副作用
        self.worker.stop(timeout=1.0)

    def test_submit_frame_drop_oldest_when_full(self):
        """
        验证队列满时自动触发【丢旧保新 (Drop-Oldest)】策略，且丢弃计数正确。
        """
        self.worker.start()
        # 暂停消费循环以便观察入队与丢弃
        self.worker._running = False

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # maxsize 为 5，连续塞入 7 帧
        self.worker._running = True  # 允许 submit
        # 通过将消费线程置为 None 模拟消费者挂起/延迟
        with patch.object(self.worker, '_worker_loop'):
            for i in range(7):
                self.worker.submit_frame(dummy_frame, timestamp=100.0 + i)

            self.assertEqual(self.worker._frame_queue.qsize(), 5, "队列最大长度必须严格保持 5，防止内存泄露！")
            self.assertGreaterEqual(self.worker.dropped_frames, 2, "多出的 2 帧必须被自动丢弃！")

    def test_async_write_frame_and_telemetry(self):
        """
        验证后台线程真实消费视频帧与遥测数据并成功落盘。
        """
        self.worker.start()

        dummy_frame = np.zeros((self.config.IMG_HEIGHT, self.config.IMG_WIDTH, 3), dtype=np.uint8)
        for i in range(3):
            self.worker.submit_frame(dummy_frame, timestamp=100.0 + i * 0.033)

        telem = TelemetryPacket(
            timestamp=100.0,
            fps=30.0,
            temp_c=45,
            total_g=1.0,
            dyn_g=0.0,
            imu_raw=[0.0, 9.8, 0.0, 0.0, 0.0, 0.0],
            target_count=1,
            lead_id=1,
            lead_dist=15.0,
            rel_speed=-1.0,
            ttc=15.0,
            alert_level="NONE",
            is_locked=False
        )
        self.worker.submit_telemetry(telem)

        # 触发紧急加锁
        self.worker.request_lock_current_segment()

        # 等待工作线程消费
        time.sleep(0.2)
        self.worker.stop(timeout=1.0)

        self.assertGreaterEqual(self.worker.written_frames, 3)
        self.assertGreaterEqual(self.worker.written_telemetry, 1)

    def test_submit_when_not_running(self):
        """
        验证未启动时提交帧与遥测被安全忽略。
        """
        self.assertFalse(self.worker._running)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = self.worker.submit_frame(dummy_frame)
        self.assertFalse(res)

        self.worker.submit_telemetry(None)
        self.assertEqual(self.worker._telemetry_queue.qsize(), 0)

    def test_submit_telemetry_queue_full(self):
        """
        验证遥测队列已满时的自动丢弃旧项并保证新项入队
        """
        self.worker._telemetry_queue = __import__('queue').Queue(maxsize=2)
        self.worker._running = True

        for i in range(3):
            t = TelemetryPacket(
                timestamp=100.0 + i, fps=30.0, temp_c=45, total_g=1.0, dyn_g=0.0,
                imu_raw=None, target_count=0, lead_id=-1, lead_dist=0.0, rel_speed=0.0,
                ttc=float("inf"), alert_level="NONE", is_locked=False
            )
            self.worker.submit_telemetry(t)

        self.assertEqual(self.worker._telemetry_queue.qsize(), 2)

    def test_consume_remaining_queue_on_stop(self):
        """
        验证在停止时，残留队列中的数据会被 _consume_remaining_queue 刷出
        """
        dummy_frame = np.zeros((self.config.IMG_HEIGHT, self.config.IMG_WIDTH, 3), dtype=np.uint8)
        self.worker._running = True
        self.worker.submit_frame(dummy_frame)
        self.worker.submit_telemetry(TelemetryPacket(
            timestamp=100.0, fps=30.0, temp_c=45, total_g=1.0, dyn_g=0.0,
            imu_raw=None, target_count=0, lead_id=-1, lead_dist=0.0, rel_speed=0.0,
            ttc=float("inf"), alert_level="NONE", is_locked=False
        ))
        self.worker.request_lock_current_segment()
        # 直接调用停止清理
        self.worker.stop(timeout=0.5)
        self.assertGreaterEqual(self.worker.written_frames, 1)
        self.assertGreaterEqual(self.worker.written_telemetry, 1)


if __name__ == '__main__':
    unittest.main()
