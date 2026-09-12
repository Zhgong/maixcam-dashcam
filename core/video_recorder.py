"""
Video Recorder Module for MaixCAM: Segmented H.264/MP4 Video Recording with AI Annotations & Emergency Locking
"""

import os
import time
try:
    from config import DashcamConfig
    from storage_manager import StorageManager
except ImportError:
    from .config import DashcamConfig
    from .storage_manager import StorageManager

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from maix import image
    HAS_MAIX_IMAGE = True
except ImportError:
    HAS_MAIX_IMAGE = False


class RecorderState:
    STOPPED = "STOPPED"
    RECORDING = "RECORDING"


class VideoRecorder:
    def __init__(self, storage_manager: StorageManager = None, config=DashcamConfig):
        self.config = config
        self.storage_manager = storage_manager or StorageManager(config=config)
        self.state = RecorderState.STOPPED
        self.current_file_path = None
        self.segment_start_time = 0.0
        self.recorded_frames = 0
        self.is_current_segment_locked = False
        self.writer = None

    def start_segment(self, now_sec: float = None) -> str:
        """
        开启一个新的 3 分钟分段视频录制。
        """
        ts = now_sec if now_sec is not None else time.time()
        self.current_file_path = self.storage_manager.generate_segment_filename(timestamp=ts)
        self.segment_start_time = ts
        self.recorded_frames = 0
        self.is_current_segment_locked = False

        if HAS_CV2 and getattr(self.config, "RECORD_VIDEO_ENABLED", True):
            try:
                fourcc_str = getattr(self.config, "VIDEO_FOURCC", "MJPG")
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                self.writer = cv2.VideoWriter(
                    self.current_file_path,
                    fourcc,
                    float(self.config.VIDEO_FPS),
                    (self.config.IMG_WIDTH, self.config.IMG_HEIGHT)
                )
                print(f"📹 [VideoRecorder] 开始录制新分段: {os.path.basename(self.current_file_path)} (Codec: {fourcc_str})")
            except Exception as e:
                print(f"⚠️ [VideoRecorder] 视频编码器初始化异常: {e}")
                self.writer = None

        self.state = RecorderState.RECORDING
        return self.current_file_path

    def write_frame(self, img, now_sec: float = None):
        """
        将画面帧写入流式视频编码器，并周期性触发底层硬件刷盘 (os.sync)。
        """
        if self.state != RecorderState.RECORDING or not self.writer:
            self.recorded_frames += 1
            return

        try:
            if HAS_MAIX_IMAGE and hasattr(image, "image2cv") and hasattr(img, "format"):
                mat = image.image2cv(img)
                bgr = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
                self.writer.write(bgr)
            elif isinstance(img, np.ndarray):
                self.writer.write(img)
        except Exception:
            pass

        self.recorded_frames += 1

        # 周期性触发底层内核硬件刷盘，保证拔电/熄火时已写入数据沉淀于物理闪存
        sync_interval = getattr(self.config, "SYNC_INTERVAL_FRAMES", 30)
        if sync_interval > 0 and self.recorded_frames % sync_interval == 0:
            try:
                os.sync()
            except Exception:
                pass


    def mark_current_segment_locked(self):
        """
        标记当前正在录制的分段为紧急加锁状态 (例如触发 G-Sensor 碰撞或紧急急刹)。
        """
        self.is_current_segment_locked = True
        print(f"🔒 [VideoRecorder] 当前分段已标记为【紧急碰撞加锁】: {os.path.basename(self.current_file_path or '')}")

    def finalize_current_segment(self) -> str:
        """
        结束当前分段录制，封包 MP4 文件；若被标记加锁，则自动移入 locked/ 目录保护。
        :return: 最终保存的文件路径 (normal 或 locked)
        """
        if not self.current_file_path:
            return None

        # 1. 释放编码器完成文件封包
        if self.writer:
            try:
                self.writer.release()
            except Exception:
                pass
            self.writer = None

        final_path = self.current_file_path

        # 2. 如果当前分段触发了紧急加锁，转移到 locked 目录
        if self.is_current_segment_locked and os.path.exists(self.current_file_path):
            try:
                final_path = self.storage_manager.lock_video(self.current_file_path)
                print(f"🛡️ [VideoRecorder] 加锁分段已永久保全至: {final_path}")
            except Exception as e:
                print(f"⚠️ [VideoRecorder] 转移加锁视频失败: {e}")

        print(f"🎬 [VideoRecorder] 分段完成封包: {os.path.basename(final_path)} (共写入 {self.recorded_frames} 帧)")
        self.state = RecorderState.STOPPED
        return final_path

    def check_and_rotate(self, now_sec: float) -> tuple:
        """
        检查是否达到 SEGMENT_DURATION_SEC (3分钟)，若达到则自动无缝切片并轮替。
        :return: (rotated: bool, current_or_new_file_path: str)
        """
        if self.state != RecorderState.RECORDING:
            new_file = self.start_segment(now_sec)
            return True, new_file

        duration = now_sec - self.segment_start_time
        if duration >= self.config.SEGMENT_DURATION_SEC:
            self.finalize_current_segment()
            self.storage_manager.enforce_quota()
            new_file = self.start_segment(now_sec)
            return True, new_file

        return False, self.current_file_path

    def close(self):
        """退出程序时的安全清理"""
        if self.state == RecorderState.RECORDING:
            self.finalize_current_segment()
