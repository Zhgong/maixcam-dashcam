"""
Storage Manager for Dashcam: Circular Video Buffer & Emergency File Locking
"""

import os
import shutil
import time
try:
    from config import DashcamConfig
except ImportError:
    from .config import DashcamConfig



class StorageManager:
    def __init__(self, base_dir=None, config=DashcamConfig):
        self.config = config
        self.base_dir = base_dir or config.DEFAULT_STORAGE_BASE
        self.normal_dir = os.path.join(self.base_dir, "normal")
        self.locked_dir = os.path.join(self.base_dir, "locked")
        self.snapshots_dir = os.path.join(self.base_dir, "snapshots")

        self.ensure_directories()

    def ensure_directories(self):
        """确保存储根目录与各个子目录存在"""
        os.makedirs(self.normal_dir, exist_ok=True)
        os.makedirs(self.locked_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)

    def generate_segment_filename(self, timestamp: float = None) -> str:
        """
        生成规范的分段视频文件完整路径 (normal/YYYYMMDD_HHMMSS.avi 或 .mp4)。
        """
        ts = timestamp if timestamp is not None else time.time()
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
        ext = getattr(self.config, "VIDEO_CONTAINER_EXT", ".avi")
        filename = f"{time_str}{ext}"
        return os.path.join(self.normal_dir, filename)

    def lock_video(self, file_path: str) -> str:
        """
        将 normal 目录下的录像文件移动到 locked 目录进行永久写保护。
        :param file_path: normal 目录下的原始文件路径
        :return: locked 目录下的新文件路径
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")

        filename = os.path.basename(file_path)
        dest_path = os.path.join(self.locked_dir, filename)

        # 原子移动到 locked 目录
        shutil.move(file_path, dest_path)
        return dest_path

    def get_normal_files(self) -> list:
        """
        获取 normal 目录下所有视频文件 (.avi / .mp4)，并按修改时间从旧到新 (升序) 排序。
        """
        if not os.path.exists(self.normal_dir):
            return []

        files = [
            os.path.join(self.normal_dir, f)
            for f in os.listdir(self.normal_dir)
            if f.endswith((".avi", ".mp4"))
        ]
        # 按修改时间从早到晚排序
        files.sort(key=lambda p: os.path.getmtime(p))
        return files


    def enforce_quota(self, max_normal_files: int = None) -> int:
        """
        执行 FIFO 循环空间配额清理：
        当 normal 文件数超过阈值时，自动删除最旧的文件。
        注意：locked 目录下的文件绝不参与清理！
        :return: 实际清理的文件数
        """
        max_files = max_normal_files if max_normal_files is not None else self.config.MAX_NORMAL_FILES
        normal_files = self.get_normal_files()

        deleted_count = 0
        if len(normal_files) > max_files:
            to_delete_count = len(normal_files) - max_files
            for i in range(to_delete_count):
                file_to_del = normal_files[i]
                try:
                    os.remove(file_to_del)
                    deleted_count += 1
                except OSError as e:
                    print(f"[StorageManager] 清理旧视频失败 {file_to_del}: {e}")

        return deleted_count
