"""
Offline Replay Player & Simulator for MaixCAM Dashcam
Renders clean raw road video with real-time ADAS HUD overlay (Virtual Lane Corridor, Distance & Alerts)
"""

import os
import sys
import argparse
import time
import math
import cv2
import pandas as pd
import numpy as np

# 添加 core 路径
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if proj_dir not in sys.path:
    sys.path.insert(0, proj_dir)

from core.config import DashcamConfig
from core.fcw_tracker import FCWAnalyzer, AlertLevel
from core.hud_renderer import HUDRenderer
from core.road_detector import RoadDetector, RoadDetectionResult


class MockMaixImage:
    """包装 OpenCV 图像以适配 HUDRenderer 绘图接口"""
    def __init__(self, cv_bgr):
        self.bgr = cv_bgr

    def draw_rect(self, x, y, w, h, color, thickness=1):
        # 兼容 RGB / BGR tuple
        c = (color[2], color[1], color[0]) if isinstance(color, (tuple, list)) and len(color) >= 3 else color
        cv2.rectangle(self.bgr, (int(x), int(y)), (int(x + w), int(y + h)), c, int(thickness))

    def draw_line(self, x1, y1, x2, y2, color, thickness=1):
        # 兼容 RGB / BGR tuple
        c = (color[2], color[1], color[0]) if isinstance(color, (tuple, list)) and len(color) >= 3 else color
        cv2.line(self.bgr, (int(x1), int(y1)), (int(x2), int(y2)), c, int(thickness), cv2.LINE_AA)

    def draw_string(self, x, y, text, color, scale=1.0):
        c = (color[2], color[1], color[0]) if isinstance(color, (tuple, list)) and len(color) >= 3 else color
        cv2.putText(self.bgr, text, (int(x), int(y + 14 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.45, c, 1, cv2.LINE_AA)


def find_associated_telemetry(video_path: str) -> pd.DataFrame:
    """自动匹配对应行程的遥测 CSV 数据"""
    v_base = os.path.basename(video_path).replace(".avi", "").replace(".mp4", "")
    # 查找目录
    data_dir = os.path.dirname(os.path.dirname(video_path))
    telem_dir = os.path.join(data_dir, "telemetry")
    if not os.path.exists(telem_dir):
        return None

    # 尝试匹配时间戳
    import time
    try:
        v_time = time.mktime(time.strptime(v_base, "%Y%m%d_%H%M%S"))
    except Exception:
        return None

    best_csv = None
    min_diff = float("inf")
    for c in os.listdir(telem_dir):
        if not c.endswith(".csv"):
            continue
        c_path = os.path.join(telem_dir, c)
        try:
            df = pd.read_csv(c_path).dropna(subset=['timestamp'])
            if len(df) > 0:
                t0 = df['timestamp'].iloc[0]
                t1 = df['timestamp'].iloc[-1]
                if t0 - 60 <= v_time <= t1 + 60:
                    return df
                diff = abs(t0 - v_time)
                if diff < min_diff and diff < 300:
                    min_diff = diff
                    best_csv = df
        except Exception:
            pass

    return best_csv


def run_replay(video_path: str, save_annotated: str = None, display_window: bool = True):
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return

    print(f"🎬 [Replay] 正在加载视频: {os.path.basename(video_path)}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  • 视频属性: {w}x{h} @ {fps:.1f} FPS, 共 {total_frames} 帧 ({total_frames/fps:.1f} 秒)")

    # 自动关联遥测数据
    df_telem = find_associated_telemetry(video_path)
    if df_telem is not None:
        print(f"  • 已成功对齐路测时序遥测数据 (共 {len(df_telem)} 行)")
    else:
        print("  • 未找到精确对应的时序遥测，将采用合成 ADAS HUD 推演模式")

    config = DashcamConfig
    analyzer = FCWAnalyzer(config=config)
    hud = HUDRenderer(config=config)

    # 加载本地 YOLO11 视觉检测器
    from tools.yolo_detector import YOLO11Detector
    detector = YOLO11Detector()
    if detector.is_ready:
        print("🧠 [Replay] 本地 YOLO11 视觉检测引擎已成功挂载！")
    else:
        print("⚠️ [Replay] 未检测到本地 YOLO11 权重，将回退至遥测合成推演模式。")

    writer = None
    if save_annotated:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG") if save_annotated.endswith(".avi") else cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_annotated, fourcc, fps, (w, h))
        print(f"📹 [Replay] 渲染后的带 HUD 视频将保存至: {save_annotated}")

    # 支持窗口交互回放
    window_name = f"Camp Dashcam v1.2.0 Offline Replay - {os.path.basename(video_path)}"
    if display_window and os.environ.get("DISPLAY"):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)

    frame_idx = 0
    t0 = time.time()

    road_detector = RoadDetector(config=config)

    # 自动检查安装朝向 (从遥测或参数判断是否需要画面纠正)
    is_inverted = False
    if df_telem is not None and 'acc_y' in df_telem.columns:
        mean_acc_y = df_telem['acc_y'].mean()
        if analyzer.is_camera_inverted(mean_acc_y):
            is_inverted = True
            print(f"🔄 [Replay] 检测到该段录像为倒吊安装 (acc_y均值 {mean_acc_y:.2f} < -3.0)，自动执行 180° 正向纠偏！")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 若倒装录制的原始视频，自动执行 180 度翻转以恢复正向视觉
        if is_inverted:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        now_sec = t0 + (frame_idx / fps)
        detections = []

        yaw_rate = 0.0
        horizon_y = analyzer.y_center
        if df_telem is not None and frame_idx < len(df_telem):
            row = df_telem.iloc[frame_idx]
            yaw_rate = float(row.get('gyro_y', 0.0))

            if 'acc_y' in row and 'acc_z' in row:
                ay = float(row['acc_y'])
                az = float(row['acc_z'])
                ax = float(row.get('acc_x', 0.0))
                # 倒装坐标系归一化
                ax_eff = -ax if is_inverted else ax
                ay_eff = -ay if is_inverted else ay
                az_eff = az
                pitch_deg = analyzer.compute_pitch_from_gravity(ax_eff, ay_eff, az_eff)
                horizon_y = analyzer.compute_horizon_y(pitch_deg)
                analyzer.current_pitch_deg = pitch_deg
                analyzer.current_horizon_y = horizon_y

        # 真实道路边界与车道标线识别
        road_res = road_detector.detect_actual_road(frame, horizon_y=int(horizon_y), imu_yaw_rate=yaw_rate)

        # 优先使用本地 YOLO11 对正向画面执行前向推理
        if detector.is_ready:
            detections = detector.detect(frame)
        elif df_telem is not None and frame_idx < len(df_telem):
            row = df_telem.iloc[frame_idx]
            dist = float(row.get('lead_dist', 0.0))
            if dist > 0.5:
                total_angle = math.atan(config.CAMERA_HEIGHT_M / dist)
                dy = math.tan(total_angle - math.radians(config.CAMERA_PITCH_DEG)) * analyzer.fy
                bottom_y = analyzer.y_center + dy
                box_w = max(40, min(300, int(analyzer.fx * 1.8 / dist)))
                box_h = max(30, min(240, int(analyzer.fy * 1.5 / dist)))
                box_x = max(10, min(w - box_w - 10, int(analyzer.x_center - box_w / 2.0)))
                box_y = max(10, min(h - box_h - 10, int(bottom_y - box_h)))
                detections.append({
                    'track_id': int(row.get('lead_id', 1)),
                    'class_id': config.TARGET_CAR,
                    'bbox': [box_x, box_y, box_w, box_h]
                })

        alerts = analyzer.process_detections(detections, now_sec=now_sec, yaw_rate_dps=yaw_rate)

        # 封装并渲染 HUD (包含实际物理道路多段线与动态地平线)
        mock_img = MockMaixImage(frame)
        hud.render_hud(
            img=mock_img,
            tracks=analyzer.tracks,
            alerts=alerts,
            is_recording=True,
            is_locked=False,
            rec_seconds=int(frame_idx / fps),
            temp_c=58,
            total_g=1.0,
            snap_count=12,
            yaw_rate_dps=yaw_rate,
            road_res=road_res,
            horizon_y=int(horizon_y)
        )


        if writer:
            writer.write(frame)

        if display_window and os.environ.get("DISPLAY"):
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(int(1000 / fps)) & 0xFF
            if key == 27 or key == ord('q'):  # ESC 或 q 退出
                print("⏹️ 用户中断回放")
                break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    if display_window and os.environ.get("DISPLAY"):
        cv2.destroyAllWindows()

    print(f"✅ 回放完成！共处理 {frame_idx} 帧画面。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dashcam Offline Replay Player & Simulator")
    parser.add_argument("--video", type=str, help="Path to raw video (.avi / .mp4)")
    parser.add_argument("--save", type=str, default=None, help="Save annotated video with HUD")
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI window popup")
    args = parser.parse_args()

    if args.video:
        run_replay(args.video, save_annotated=args.save, display_window=not args.no_gui)
    else:
        # 默认使用第一段有路况的视频
        def_v = "./data/road_test_trip3_20260829/locked/20260829_181714.avi"
        run_replay(def_v, save_annotated=None, display_window=True)
