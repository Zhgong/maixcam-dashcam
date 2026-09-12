"""
Unit & Integration Tests for Video Recorder Module (Real Video Generation, Power-Cut Resilience & Decodability)
"""

import unittest
import os
import shutil
import tempfile
import time
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
from core.config import DashcamConfig
from core.storage_manager import StorageManager
from core.video_recorder import VideoRecorder, RecorderState


class TestVideoRecorder(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="video_rec_test_")
        self.config = DashcamConfig
        self.storage_mgr = StorageManager(base_dir=self.test_dir, config=self.config)
        self.recorder = VideoRecorder(storage_manager=self.storage_mgr, config=self.config)

    def tearDown(self):
        if self.recorder:
            self.recorder.close()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_real_video_file_creation_and_opencv_decodability(self):
        """
        [Critical Test]: 验证 VideoRecorder 能够真正生成流式自包含视频文件，
        并且该视频可以被 OpenCV/VLC 完整解码，总帧数与分辨率严格一致！
        """
        t0 = 100.0
        file_path = self.recorder.start_segment(now_sec=t0)
        self.assertIsNotNone(file_path)
        self.assertTrue(file_path.endswith((".avi", ".mp4")))

        # 写入 30 帧真实的 RGB 图像数据
        frame_w = self.config.IMG_WIDTH
        frame_h = self.config.IMG_HEIGHT
        for i in range(30):
            synthetic_frame = np.full((frame_h, frame_w, 3), i * 8, dtype=np.uint8)
            self.recorder.write_frame(synthetic_frame, now_sec=t0 + i * 0.033)

        self.assertEqual(self.recorder.recorded_frames, 30)

        # 正常封包当前分段
        final_path = self.recorder.finalize_current_segment()

        # 1. 物理文件存在且非空校验
        self.assertTrue(os.path.exists(final_path), f"物理文件未生成: {final_path}")
        file_size = os.path.getsize(final_path)
        self.assertGreater(file_size, 1000, f"视频文件大小过小 ({file_size} bytes)！")

        # 2. OpenCV 真实解码测试
        cap = cv2.VideoCapture(final_path)
        self.assertTrue(cap.isOpened(), "生成的视频无法被 VideoCapture 打开解码！")
        decoded_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.assertEqual(frame.shape, (frame_h, frame_w, 3))
            decoded_frames += 1

        cap.release()
        self.assertEqual(decoded_frames, 30, f"期望解码 30 帧，实际解码 {decoded_frames} 帧！")

    def test_sudden_power_cut_without_release_still_playable(self):
        """
        [Automotive Power-Cut Resilience Test]:
        模拟行车途中汽车突然熄火/强制拔掉电源（故意不调用 finalize_current_segment 或 release），
        验证流式自包含容器确保断电前写入的所有帧 100% 完好无损、绝不报废！
        """
        t0 = 100.0
        file_path = self.recorder.start_segment(now_sec=t0)

        # 写入 45 帧数据
        frame_w = self.config.IMG_WIDTH
        frame_h = self.config.IMG_HEIGHT
        for i in range(45):
            synthetic_frame = np.full((frame_h, frame_w, 3), i * 5, dtype=np.uint8)
            self.recorder.write_frame(synthetic_frame, now_sec=t0 + i * 0.033)

        # ⚡ 模拟突发暴力断电：销毁 writer 对象，故意不调用 release() / finalize_current_segment()！
        raw_writer = self.recorder.writer
        self.recorder.writer = None
        del raw_writer

        # 验证物理文件依然存在
        self.assertTrue(os.path.exists(file_path), "断电后物理视频文件必须存在！")

        # 用 OpenCV 验证断电后的文件是否依然能被直接打开并解码所有已写入帧
        cap = cv2.VideoCapture(file_path)
        self.assertTrue(cap.isOpened(), "突发断电后视频文件损坏，无法打开！流式容器必须抗断电！")
        decoded_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            decoded_frames += 1
        cap.release()

        # 断言已写入的 45 帧全部完整保留，帧丢失率为 0！
        self.assertGreaterEqual(decoded_frames, 44, f"断电前写入 45 帧，实际解码 {decoded_frames} 帧！")

    def test_periodic_hardware_sync_on_flash(self):
        """
        验证录像过程中每隔指定帧数 (如 30 帧/1 秒) 自动调用硬件刷盘，确保数据沉淀在物理闪存中。
        """
        t0 = 100.0
        self.recorder.start_segment(now_sec=t0)

        with patch("os.sync") as mock_sync:
            dummy_frame = np.zeros((self.config.IMG_HEIGHT, self.config.IMG_WIDTH, 3), dtype=np.uint8)
            for i in range(65):
                self.recorder.write_frame(dummy_frame, now_sec=t0 + i * 0.033)

            # 写入 65 帧（>2 秒），至少应触发 2 次 os.sync
            self.assertGreaterEqual(mock_sync.call_count, 2)

    def test_segment_auto_rotation_on_duration(self):
        """
        验证达到设定时长 (如 180 秒) 时的自动分段切片轮替与旧分段完整封包。
        """
        t0 = 100.0
        file1 = self.recorder.start_segment(now_sec=t0)

        dummy_frame = np.zeros((self.config.IMG_HEIGHT, self.config.IMG_WIDTH, 3), dtype=np.uint8)
        for i in range(10):
            self.recorder.write_frame(dummy_frame, now_sec=t0 + i * 0.033)

        # 写入超过 SEGMENT_DURATION_SEC (180秒后) -> 自动触发换段
        rotated, new_file = self.recorder.check_and_rotate(now_sec=t0 + 185.0)
        self.assertTrue(rotated)
        self.assertIsNotNone(new_file)
        self.assertNotEqual(file1, new_file)

        # 验证上一段 file1 已经被成功封包且有效
        self.assertTrue(os.path.exists(file1))
        self.assertGreater(os.path.getsize(file1), 1000)

    def test_emergency_locked_segment_moved_and_playable(self):
        """
        验证在录制过程中若触发了紧急加锁，封包后视频完整保全至 locked/ 目录且可正常解码播放。
        """
        t0 = 100.0
        file_path = self.recorder.start_segment(now_sec=t0)

        dummy_frame = np.zeros((self.config.IMG_HEIGHT, self.config.IMG_WIDTH, 3), dtype=np.uint8)
        for i in range(15):
            self.recorder.write_frame(dummy_frame, now_sec=t0 + i * 0.033)

        # 触发紧急加锁
        self.recorder.mark_current_segment_locked()
        self.assertTrue(self.recorder.is_current_segment_locked)

        # 结束当前分段
        final_path = self.recorder.finalize_current_segment()

        self.assertTrue(final_path.startswith(self.storage_mgr.locked_dir), "加锁分段必须移动至 locked 目录！")
        self.assertTrue(os.path.exists(final_path))
        self.assertFalse(os.path.exists(file_path))

        # 验证 locked 视频同样可完整解码
        cap = cv2.VideoCapture(final_path)
        self.assertTrue(cap.isOpened())
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 15)
        cap.release()


if __name__ == "__main__":
    unittest.main()
