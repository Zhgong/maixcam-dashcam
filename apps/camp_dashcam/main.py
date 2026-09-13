#!/usr/bin/env python3
"""
MaixCAM 2 AI Dashcam & Forward Collision Warning (FCW) Main Application
Refactored for Decoupled Multi-Pipeline Architecture (Issue #2)
"""

import sys
import os
import time
import signal

app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

try:
    from config import DashcamConfig
    from pipeline_manager import PipelineManager
except ImportError:
    try:
        from core.config import DashcamConfig
        from core.pipeline_manager import PipelineManager
    except ImportError:
        from .config import DashcamConfig
        from .pipeline_manager import PipelineManager

# 尝试导入硬件库 (板端真实运行 vs 本地 Mock)
try:
    from maix import camera, display, nn, image, time as mtime, app, touchscreen, ext_dev
    IS_HARDWARE = True
    disp = display.Display()
except ImportError:
    IS_HARDWARE = False
    disp = None


def parse_touch_exit(touch_data, img_w=640, img_h=480) -> bool:
    """
    解析 MaixPy 触控输入 [x, y, pressed]，判断是否点击了退出区域。
    """
    if not touch_data or len(touch_data) < 3:
        return False

    x, y, pressed = touch_data[0], touch_data[1], touch_data[2]
    if not pressed:
        return False

    # 触碰四个角边缘 Exit 区域
    if (y >= img_h - 60 or y <= 60):
        if (x >= img_w - 140 or x <= 140):
            return True

    return False


def main(disp_handle=None):
    active_disp = disp_handle or disp
    print(f"🚗 [Camp Dashcam] 启动车载 AI 记录仪与碰撞预警 ({DashcamConfig.APP_VERSION})...")

    # 1. 硬件外设初始化
    cam = None
    detector = None
    imu_sensor = None
    ts = None

    if IS_HARDWARE:
        try:
            detector = nn.YOLO11(model='/root/models/yolo11n.mud')
            print("🧠 [Camp Dashcam] YOLO11 NPU 视觉模型加载完成！")
        except Exception as e:
            print(f"⚠️ [Camp Dashcam] NPU 模型加载失败 (降级运行): {e}")

        try:
            cam = camera.Camera(DashcamConfig.IMG_WIDTH, DashcamConfig.IMG_HEIGHT, image.Format.FMT_RGB888, fps=30)
            print("📷 [Camp Dashcam] 摄像头 30FPS 初始化完成！")
        except Exception as e:
            print(f"⚠️ [Camp Dashcam] 摄像头初始化异常: {e}")

        try:
            imu_sensor = ext_dev.imu.IMU("lsm6dsowtr")
            print("🧭 [Camp Dashcam] LSM6DSOWTR 6轴 IMU 传感器已就绪！")
            if active_disp:
                active_disp.set_vflip(True)
                active_disp.set_hmirror(False)
        except Exception as e:
            print(f"⚠️ [Camp Dashcam] IMU 初始化异常: {e}")

        try:
            ts = touchscreen.TouchScreen()
        except Exception:
            ts = None

    # 2. 构造并启动四流水线中枢调度器
    pipeline = PipelineManager(
        config=DashcamConfig,
        camera_dev=cam,
        detector_dev=detector,
        imu_dev=imu_sensor,
        display_dev=active_disp,
        touch_dev=ts
    )

    # 优雅退出信号处理
    def handle_signal(sig, frame):
        print(f"👋 [Camp Dashcam] 捕获退出信号 ({sig})，正在平稳关闭所有流水线...")
        pipeline.stop()
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    except Exception:
        pass

    pipeline.start()
    print("🚀 [Camp Dashcam] 四流水线 (Perception / Kinematics / Display / Storage) 已全部并发运转！")

    # 3. 主控制监听循环 (响应触摸屏与硬件 app 退出事件)
    try:
        while pipeline._running:
            if IS_HARDWARE and app.need_exit():
                break

            if ts:
                touch_data = ts.read()
                if parse_touch_exit(touch_data, DashcamConfig.IMG_WIDTH, DashcamConfig.IMG_HEIGHT):
                    print("👋 [Camp Dashcam] 接收到退出触摸手势，平稳退出...")
                    break

            if not IS_HARDWARE:
                # 本地环境单次启动后测试完毕退出
                time.sleep(0.1)
                break

            time.sleep(0.05)
    finally:
        pipeline.stop()
        print("🛑 [Camp Dashcam] App 已安全退出并完成视频封包、遥测存盘与资源释放。")


if __name__ == "__main__":
    main()
