"""
Async Storage Worker for MaixCAM Dashcam: Decoupled Video Recording, Telemetry & Flash I/O
Eliminates camera preview & UI frame drops caused by SD/eMMC flash write latency and os.sync()
"""

import threading
import queue
import time
from typing import Optional, Dict, Any

try:
    from config import DashcamConfig
    from storage_manager import StorageManager
    from video_recorder import VideoRecorder
    from telemetry_logger import TelemetryLogger
except ImportError:
    from .config import DashcamConfig
    from .storage_manager import StorageManager
    from .video_recorder import VideoRecorder
    from .telemetry_logger import TelemetryLogger


class RecordFramePacket:
    """封装待异步写入的视频帧数据包"""
    __slots__ = ("image", "timestamp")

    def __init__(self, image: Any, timestamp: float):
        self.image = image
        self.timestamp = timestamp


class TelemetryPacket:
    """封装待异步追加的结构化遥测数据包"""
    __slots__ = (
        "timestamp", "fps", "temp_c", "total_g", "dyn_g", "imu_raw",
        "target_count", "lead_id", "lead_dist", "rel_speed", "ttc",
        "alert_level", "is_locked"
    )

    def __init__(
        self,
        timestamp: float,
        fps: float,
        temp_c: int,
        total_g: float,
        dyn_g: float,
        imu_raw: Optional[list],
        target_count: int,
        lead_id: int,
        lead_dist: float,
        rel_speed: float,
        ttc: float,
        alert_level: str,
        is_locked: bool
    ):
        self.timestamp = timestamp
        self.fps = fps
        self.temp_c = temp_c
        self.total_g = total_g
        self.dyn_g = dyn_g
        self.imu_raw = imu_raw
        self.target_count = target_count
        self.lead_id = lead_id
        self.lead_dist = lead_dist
        self.rel_speed = rel_speed
        self.ttc = ttc
        self.alert_level = alert_level
        self.is_locked = is_locked


class AsyncStorageWorker:
    """
    异步存储与遥测工作线程：
    1. 维护有界且丢旧保新的视频缓冲队列 (maxsize=30，约 1 秒视频缓冲)；
    2. 后台循环消费帧数据并写入磁盘分段文件，隔离底层 os.sync() 物理刷盘抖动；
    3. 维护遥测队列并执行批量写入与 flush；
    4. 自动处理 3 分钟分段轮替与存储配额清理。
    """

    def __init__(
        self,
        storage_manager: Optional[StorageManager] = None,
        video_recorder: Optional[VideoRecorder] = None,
        telemetry_logger: Optional[TelemetryLogger] = None,
        config=DashcamConfig,
        queue_maxsize: int = 30
    ):
        self.config = config
        self.storage_manager = storage_manager or StorageManager(config=config)
        self.video_recorder = video_recorder or VideoRecorder(storage_manager=self.storage_manager, config=config)
        self.telemetry_logger = telemetry_logger or TelemetryLogger(config=config)

        self.queue_maxsize = queue_maxsize
        self._frame_queue = queue.Queue(maxsize=queue_maxsize)
        self._telemetry_queue = queue.Queue(maxsize=100)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock_request_flag = False

        # 统计指标
        self.dropped_frames = 0
        self.written_frames = 0
        self.written_telemetry = 0

    def start(self):
        """启动后台工作线程"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, name="AsyncStorageWorker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0):
        """优雅停止后台工作线程并封包文件"""
        if not self._running:
            return
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        else:
            self._consume_remaining_queue()
        self._flush_and_close()

    def submit_frame(self, image: Any, timestamp: Optional[float] = None) -> bool:
        """
        提交一帧图像供后台录像。
        若队列已满，采取【丢旧保新 (Drop-Oldest)】策略，弹出最老的一帧，保证内存绝对不膨胀。
        :return: bool (True=成功放入, False=丢弃旧帧后放入)
        """
        if not self._running or image is None:
            return False

        ts = timestamp if timestamp is not None else time.time()
        packet = RecordFramePacket(image=image, timestamp=ts)

        dropped = False
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
                self.dropped_frames += 1
                dropped = True
            except queue.Empty:
                pass

        self._frame_queue.put_nowait(packet)
        return not dropped

    def submit_telemetry(self, packet: TelemetryPacket):
        """提交一帧结构化遥测数据包"""
        if not self._running or packet is None:
            return
        if self._telemetry_queue.full():
            try:
                self._telemetry_queue.get_nowait()
            except queue.Empty:
                pass
        self._telemetry_queue.put_nowait(packet)

    def request_lock_current_segment(self):
        """请求将当前分段紧急加锁 (如触发 G-Sensor 碰撞或紧急急刹)"""
        self._lock_request_flag = True

    def _worker_loop(self):
        """后台消费者主循环"""
        # 初始启动视频分段
        now_sec = time.time()
        self.video_recorder.start_segment(now_sec=now_sec)

        while self._running:
            # 1. 检查是否有加锁请求
            if self._lock_request_flag:
                self.video_recorder.mark_current_segment_locked()
                self._lock_request_flag = False

            # 2. 消费视频帧队列 (带超时等待，避免忙轮询)
            had_work = False
            try:
                frame_pkt: RecordFramePacket = self._frame_queue.get(timeout=0.03)
                self.video_recorder.write_frame(frame_pkt.image, now_sec=frame_pkt.timestamp)
                self.written_frames += 1
                had_work = True
                self._frame_queue.task_done()
            except queue.Empty:
                pass

            # 3. 批量消费遥测队列
            while True:
                try:
                    telem_pkt: TelemetryPacket = self._telemetry_queue.get_nowait()
                    self.telemetry_logger.log_frame(
                        timestamp=telem_pkt.timestamp,
                        fps=telem_pkt.fps,
                        temp_c=telem_pkt.temp_c,
                        total_g=telem_pkt.total_g,
                        dyn_g=telem_pkt.dyn_g,
                        imu_raw=telem_pkt.imu_raw,
                        target_count=telem_pkt.target_count,
                        lead_id=telem_pkt.lead_id,
                        lead_dist=telem_pkt.lead_dist,
                        rel_speed=telem_pkt.rel_speed,
                        ttc=telem_pkt.ttc,
                        alert_level=telem_pkt.alert_level,
                        is_locked=telem_pkt.is_locked
                    )
                    self.written_telemetry += 1
                    had_work = True
                    self._telemetry_queue.task_done()
                except queue.Empty:
                    break

            # 4. 自动分段轮替与存储配额检查
            loop_now = time.time()
            rotated, _ = self.video_recorder.check_and_rotate(now_sec=loop_now)
            if rotated:
                self.telemetry_logger.flush()

            if not had_work:
                time.sleep(0.01)

        # 退出时清空残留队列
        self._consume_remaining_queue()

    def _consume_remaining_queue(self):
        """退出前将队列内剩余的数据写盘"""
        while not self._frame_queue.empty():
            try:
                pkt = self._frame_queue.get_nowait()
                self.video_recorder.write_frame(pkt.image, now_sec=pkt.timestamp)
                self.written_frames += 1
                self._frame_queue.task_done()
            except queue.Empty:
                break

        while not self._telemetry_queue.empty():
            try:
                telem = self._telemetry_queue.get_nowait()
                self.telemetry_logger.log_frame(
                    timestamp=telem.timestamp, fps=telem.fps, temp_c=telem.temp_c,
                    total_g=telem.total_g, dyn_g=telem.dyn_g, imu_raw=telem.imu_raw,
                    target_count=telem.target_count, lead_id=telem.lead_id,
                    lead_dist=telem.lead_dist, rel_speed=telem.rel_speed, ttc=telem.ttc,
                    alert_level=telem.alert_level, is_locked=telem.is_locked
                )
                self.written_telemetry += 1
                self._telemetry_queue.task_done()
            except queue.Empty:
                break

    def _flush_and_close(self):
        """封包当前录像与关闭日志"""
        if self._lock_request_flag:
            self.video_recorder.mark_current_segment_locked()
            self._lock_request_flag = False
        self.video_recorder.close()
        self.telemetry_logger.flush()
        self.telemetry_logger.close()
