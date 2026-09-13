"""
Unit tests for Dashcam Storage Manager (Circular Buffer & Video Locking)
"""

import unittest
import os
import shutil
import tempfile
import time
from core.config import DashcamConfig
from core.storage_manager import StorageManager


class TestStorageManager(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="dashcam_test_storage_")
        self.config = DashcamConfig
        self.manager = StorageManager(base_dir=self.test_dir, config=self.config)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_directory_initialization(self):
        """
        验证存储目录自动初始化：normal/、locked/、snapshots/ 必须齐备。
        """
        self.assertTrue(os.path.exists(self.manager.normal_dir))
        self.assertTrue(os.path.exists(self.manager.locked_dir))
        self.assertTrue(os.path.exists(self.manager.snapshots_dir))

    def test_generate_video_filename(self):
        """
        验证分段视频文件名生成格式 (YYYYMMDD_HHMMSS.avi 或 .mp4)。
        """
        filename = self.manager.generate_segment_filename(timestamp=1700000000)
        self.assertTrue(filename.endswith((".avi", ".mp4")))
        self.assertIn("normal", filename)


    def test_lock_current_file(self):
        """
        验证紧急加锁：将指定 normal/ 下的文件原子移动到 locked/ 目录。
        """
        # 创建一个模拟的 normal 录像文件
        test_video = os.path.join(self.manager.normal_dir, "20260829_120000.mp4")
        with open(test_video, "w") as f:
            f.write("dummy video data")

        locked_path = self.manager.lock_video(test_video)

        self.assertFalse(os.path.exists(test_video), "原 normal 路径下的文件应已被移走")
        self.assertTrue(os.path.exists(locked_path), "locked 目录下必须存在加锁文件")
        self.assertTrue(locked_path.startswith(self.manager.locked_dir))

    def test_fifo_circular_cleanup_preserves_locked(self):
        """
        验证 FIFO 循环空间清理：
        当文件总数/容量超过配额时，严格按从旧到新自动清理 normal/ 文件，但绝对禁止删除 locked/ 目录下的任何文件！
        """
        # 1. 创建 2 个加锁文件
        locked_file1 = os.path.join(self.manager.locked_dir, "locked_01.mp4")
        locked_file2 = os.path.join(self.manager.locked_dir, "locked_02.mp4")
        with open(locked_file1, "w") as f: f.write("locked")
        with open(locked_file2, "w") as f: f.write("locked")

        # 2. 创建 5 个 normal 文件 (模拟从旧到新)
        normal_files = []
        for i in range(5):
            path = os.path.join(self.manager.normal_dir, f"segment_0{i}.mp4")
            with open(path, "w") as f: f.write("normal data")
            # 调整修改时间模拟先后顺序
            os.utime(path, (1000 + i * 10, 1000 + i * 10))
            normal_files.append(path)

        # 3. 设置最大保留文件数为 2，触发清理
        deleted_count = self.manager.enforce_quota(max_normal_files=2)

        self.assertEqual(deleted_count, 3, "应该清理掉最早的 3 个 normal 文件")
        # 验证最老的 3 个被删除，最新的 2 个依然保留
        self.assertFalse(os.path.exists(normal_files[0]))
        self.assertFalse(os.path.exists(normal_files[1]))
        self.assertFalse(os.path.exists(normal_files[2]))
        self.assertTrue(os.path.exists(normal_files[3]))
        self.assertTrue(os.path.exists(normal_files[4]))

        # 验证 locked 文件毫发无损
        self.assertTrue(os.path.exists(locked_file1))
        self.assertTrue(os.path.exists(locked_file2))

    def test_lock_video_non_existent_file_raises(self):
        """
        验证 lock_video 对不存在的文件抛出 FileNotFoundError
        """
        non_existent = os.path.join(self.manager.normal_dir, "ghost_video.mp4")
        with self.assertRaises(FileNotFoundError):
            self.manager.lock_video(non_existent)

    def test_get_normal_files_when_dir_missing(self):
        """
        验证当 normal 目录不存在时返回空列表
        """
        shutil.rmtree(self.manager.normal_dir)
        files = self.manager.get_normal_files()
        self.assertEqual(files, [])

    def test_enforce_quota_deletion_error_handling(self):
        """
        验证当某个文件删除遇到 OSError 时，能够优雅捕获而不崩溃并继续
        """
        from unittest.mock import patch
        path = os.path.join(self.manager.normal_dir, "to_delete.mp4")
        with open(path, "w") as f: f.write("dummy")
        with patch('os.remove', side_effect=OSError("Permission denied")):
            deleted = self.manager.enforce_quota(max_normal_files=0)
            self.assertEqual(deleted, 0)


if __name__ == '__main__':
    unittest.main()
