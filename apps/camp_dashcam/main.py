#!/usr/bin/env python3
"""
MaixCAM 2 AI Dashcam & Forward Collision Warning (FCW) Main Application
"""

import sys
import os
import time

app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from config import DashcamConfig
from fcw_tracker import FCWAnalyzer, AlertLevel, OrientationDetector, GravityDirection, BottomEdge
from g_sensor_guard import GSensorGuard, ShockEventLevel
from storage_manager import StorageManager
from hud_renderer import HUDRenderer
from telemetry_logger import TelemetryLogger, SnapshotManager
from video_recorder import VideoRecorder
from road_detector import RoadDetector, RoadDetectionResult

# 尝试导入硬件库 (板端真实运行 vs 本地 Mock)
try:
    from maix import camera, display, nn, image, time as mtime, app, touchscreen, ext_dev, audio
    IS_HARDWARE = True
    disp = display.Display()
except ImportError:
    IS_HARDWARE = False
    disp = None


def get_soc_temperature():
    """读取 SoC 芯片温度"""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(int(f.read().strip()) / 1000)
    except Exception:
        return 45


def parse_touch_exit(touch_data, img_w=640, img_h=480) -> bool:
    """
    解析 MaixPy 触控输入 [x, y, pressed]，判断是否点击了退出区域 (右下角或左下角 [Exit] 按钮)。
    """
    if not touch_data or len(touch_data) < 3:
        return False

    x, y, pressed = touch_data[0], touch_data[1], touch_data[2]
    if not pressed:
        return False

    # 触碰四个角边缘 Exit 区域 (无论正装或倒装均可可靠响应退出手势)
    if (y >= img_h - 60 or y <= 60):
        if (x >= img_w - 140 or x <= 140):
            return True

    return False


def main(disp_handle=None):
    active_disp = disp_handle or disp
    print(f"🚗 [Camp Dashcam] 启动车载 AI 记录仪与碰撞预警 ({DashcamConfig.APP_VERSION})...")

    # 1. 基础组件初始化
    config = DashcamConfig
    fcw_analyzer = FCWAnalyzer(config=config)
    orientation_detector = OrientationDetector(ema_alpha=0.3, threshold=2.5)
    g_guard = GSensorGuard(config=config)
    storage_mgr = StorageManager(config=config)
    hud_renderer = HUDRenderer(config=config)
    telemetry_logger = TelemetryLogger(config=config)
    snapshot_mgr = SnapshotManager(config=config)
    video_recorder = VideoRecorder(storage_manager=storage_mgr, config=config)
    road_detector = RoadDetector(config=config)

    # 2. 硬件外设初始化
    cam = None
    detector = None
    imu_sensor = None
    ts = None
    is_inverted = orientation_detector.is_inverted

    if IS_HARDWARE:
        try:
            detector = nn.YOLO11(model='/root/models/yolo11n.mud')
            print("🧠 [Camp Dashcam] YOLO11 NPU 视觉模型加载完成！")
        except Exception as e:
            print(f"⚠️ [Camp Dashcam] NPU 模型加载失败 (降级运行): {e}")

        try:
            cam = camera.Camera(config.IMG_WIDTH, config.IMG_HEIGHT, image.Format.FMT_RGB888, fps=30)
            print("📷 [Camp Dashcam] 摄像头 30FPS 初始化完成！")
        except Exception as e:
            print(f"⚠️ [Camp Dashcam] 摄像头初始化异常: {e}")

        try:
            imu_sensor = ext_dev.imu.IMU("lsm6dsowtr")
            print("🧭 [Camp Dashcam] LSM6DSOWTR 6轴 IMU 传感器已就绪！")

            # 初始 IMU 状态与安装朝向自检 (1. 识别重力方向，2. 确定物理底部)
            for _ in range(6):
                mtime.sleep_ms(20)
                init_imu = imu_sensor.read()
                if init_imu and len(init_imu) >= 3:
                    orientation_detector.update(init_imu[0], init_imu[1], init_imu[2])

            is_inverted = orientation_detector.is_inverted
            # LCD 面板硬件扫描方向相对于原生相机传感器需垂直翻转，开启硬件 VO set_vflip(True)
            # 杜绝任何显存软件层翻转，实现内存图像、截图与物理屏幕 100% 所见即所得 (WYSIWYG)
            if active_disp:
                active_disp.set_vflip(True)
                active_disp.set_hmirror(False)
            print(f"🧭 [Orientation] 初始重力方向: {orientation_detector.gravity_direction}, 物理底部: {orientation_detector.bottom_edge}, Inverted: {is_inverted}")
        except Exception as e:
            print(f"⚠️ [Camp Dashcam] IMU 初始化异常: {e}")


        try:
            ts = touchscreen.TouchScreen()
        except Exception:
            ts = None

    start_time = time.time()
    segment_start_time = start_time
    last_beep_time = 0.0
    frame_count = 0
    fps = 30.0
    last_fps_time = start_time

    # 开启初始分段录像
    video_recorder.start_segment(now_sec=start_time)

    print("🚀 [Camp Dashcam] 系统进入主检测与预警循环...")

    # 3. 主事件循环
    try:
        while True:
            if IS_HARDWARE and app.need_exit():
                break

            now_sec = time.time()
            rec_duration = int(now_sec - video_recorder.segment_start_time)
            frame_count += 1

            # 计算实时 FPS
            if now_sec - last_fps_time >= 1.0:
                fps = frame_count / (now_sec - last_fps_time)
                frame_count = 0
                last_fps_time = now_sec

            # A. 优先读取 IMU 传感器：1. 识别重力方向，2. 确定物理底部，动态自适应纠偏与急刹检测
            total_g = 1.0
            dyn_g = 0.0
            yaw_rate_dps = 0.0
            imu_data = None
            if imu_sensor:
                imu_data = imu_sensor.read()
                shock_event = g_guard.evaluate_imu(imu_data, now_sec=now_sec)
                if shock_event:
                    print(f"🚨 [G-Sensor] 触发急刹/碰撞加锁: {shock_event['level']}, Total: {shock_event['total_g']:.2f}G")
                    g_guard.is_locked = True
                    video_recorder.mark_current_segment_locked()

                if imu_data and len(imu_data) >= 3:
                    acc_x, acc_y, acc_z = imu_data[0], imu_data[1], imu_data[2]
                    total_g = (acc_x**2 + acc_y**2 + acc_z**2)**0.5 / 9.80665
                    dyn_g = abs(total_g - 1.0)

                    # 1. 识别重力方向，2. 确定物理底部
                    changed = orientation_detector.update(acc_x, acc_y, acc_z)
                    if changed:
                        is_inverted = orientation_detector.is_inverted
                        print(f"🔄 [Orientation] 朝向动态切换: 重力方向={orientation_detector.gravity_direction}, 物理底部={orientation_detector.bottom_edge}, Inverted={is_inverted}")

                    acc_x_eff = -acc_x if is_inverted else acc_x
                    acc_y_eff = -acc_y if is_inverted else acc_y
                    acc_z_eff = acc_z
                    fcw_analyzer.update_pitch_with_gravity(acc_x_eff, acc_y_eff, acc_z_eff)

                if imu_data and len(imu_data) >= 5:
                    yaw_rate_dps = float(imu_data[4])

            # B. 采集图像 (纯净原始摄像头图像直出，零软件翻转，与屏幕显示 1:1 所见即所得)
            img = None
            if cam:
                img = cam.read()
            if img is None:
                if IS_HARDWARE:
                    mtime.sleep_ms(10)
                    continue
                else:
                    # 本地仿真时退出
                    break

            # C. NPU YOLO 检测与目标提取 (在 100% 原生相机坐标系中直接检测)
            detections = []
            if detector and img:
                objs = detector.detect(img, conf_th=0.35, iou_th=0.45)
                for obj in objs:
                    class_id = getattr(obj, "class_id", 0)
                    bbox = [obj.x, obj.y, obj.w, obj.h]
                    track_id = getattr(obj, "track_id", getattr(obj, "id", 1))
                    detections.append({
                        "track_id": track_id,
                        "class_id": class_id,
                        "bbox": bbox
                    })

            # D. 真实物理路面识别 (车道标线与沥青路面物理边缘)
            road_res = None
            if config.ROAD_DETECTION_ENABLED and img is not None:
                cv_img = None
                try:
                    if IS_HARDWARE and hasattr(image, "image2cv"):
                        cv_img = image.image2cv(img)
                    elif hasattr(img, "bgr"):
                        cv_img = img.bgr
                    elif hasattr(img, "__array__"):
                        cv_img = np.asarray(img)
                except Exception:
                    cv_img = None

                if cv_img is not None:
                    road_res = road_detector.detect_actual_road(
                        cv_img,
                        horizon_y=int(fcw_analyzer.current_horizon_y),
                        imu_yaw_rate=yaw_rate_dps,
                        is_inverted=is_inverted
                    )

            # E. FCW 碰撞与相对逼近速度推算 (内置 EMA 滤波与动态转弯曲率补偿)
            alerts = fcw_analyzer.process_detections(detections, now_sec=now_sec, yaw_rate_dps=yaw_rate_dps, is_inverted=is_inverted)

            # F. 车辆/高价值目标自动快照 (Snapshots)
            for t_id, target in fcw_analyzer.tracks.items():
                target_alert = next((a["level"] for a in alerts if a["target"].track_id == t_id), AlertLevel.NONE)
                if snapshot_mgr.should_take_snapshot(target, alert_level=target_alert, now_sec=now_sec):
                    snapshot_mgr.save_snapshot(img, target, alert_level=target_alert, now_sec=now_sec)

            # G. 结构化行车遥测数据记录 (Telemetry CSV)
            lead_target = min(fcw_analyzer.tracks.values(), key=lambda t: t.distance_m, default=None)
            lead_id = lead_target.track_id if lead_target else -1
            lead_dist = lead_target.distance_m if lead_target else 0.0
            lead_speed = lead_target.rel_speed_mps if lead_target else 0.0
            lead_ttc = lead_target.ttc_sec if lead_target else float("inf")
            highest_alert = alerts[0]["level"] if alerts else AlertLevel.NONE

            # 遇到 CRITICAL 紧急预警时也自动加锁当前视频分段
            has_critical = any(a["level"] == AlertLevel.CRITICAL for a in alerts)
            if has_critical:
                video_recorder.mark_current_segment_locked()

            telemetry_logger.log_frame(
                timestamp=now_sec,
                fps=fps,
                temp_c=get_soc_temperature(),
                total_g=total_g,
                dyn_g=dyn_g,
                imu_raw=imu_data if imu_sensor else None,
                target_count=len(fcw_analyzer.tracks),
                lead_id=lead_id,
                lead_dist=lead_dist,
                rel_speed=lead_speed,
                ttc=lead_ttc,
                alert_level=highest_alert,
                is_locked=(g_guard.is_locked or video_recorder.is_current_segment_locked)
            )

            # H. 蜂鸣器声光告警
            if has_critical and (now_sec - last_beep_time > 0.8):
                last_beep_time = now_sec
                try:
                    if IS_HARDWARE:
                        audio.play("/maixapp/share/sounds/alarm.wav")
                except Exception:
                    pass

            # I. 录制纯净无污染的原始行车视频 (专为离线算法回放与数据闭环设计)
            video_recorder.write_frame(img, now_sec=now_sec)

            # J. 仅在屏幕显存上渲染 ADAS HUD 仪表界面 (带 AI 框、实际道路线与动态随动弯道地毯)
            hud_renderer.render_hud(
                img=img,
                tracks=fcw_analyzer.tracks,
                alerts=alerts,
                is_recording=True,
                is_locked=(g_guard.is_locked or video_recorder.is_current_segment_locked),
                rec_seconds=rec_duration,
                temp_c=get_soc_temperature(),
                total_g=total_g,
                snap_count=snapshot_mgr.snapshot_count,
                yaw_rate_dps=yaw_rate_dps,
                road_res=road_res,
                horizon_y=int(fcw_analyzer.current_horizon_y),
                acc_y=orientation_detector.smooth_ay,
                gravity_direction=orientation_detector.gravity_direction,
                is_inverted=is_inverted
            )

            if active_disp and img:
                active_disp.show(img)

            # 定期导出当前画面截图到 /tmp/dashcam_hud_screenshot.jpg 方便调试与效果确认
            if frame_count % 30 == 0 and img is not None:
                try:
                    img.save("/tmp/dashcam_hud_screenshot.jpg")
                except Exception:
                    pass


            # J. 3 分钟自动分段录像轮替与 FIFO 循环清理
            rotated, current_video = video_recorder.check_and_rotate(now_sec=now_sec)
            if rotated:
                telemetry_logger.flush()
                g_guard.reset_lock()

            # K. 触摸退出处理
            if ts:
                touch_data = ts.read()
                if parse_touch_exit(touch_data, config.IMG_WIDTH, config.IMG_HEIGHT):
                    print("👋 [Camp Dashcam] 接收到退出触摸手势，平稳安全退出...")
                    break

    finally:
        video_recorder.close()
        telemetry_logger.close()
        print("🛑 [Camp Dashcam] App 已安全退出并完成视频封包、遥测存盘与资源释放。")


if __name__ == "__main__":
    main()
