"""
Configuration module for MaixCAM 2 AI Dashcam & FCW Guard
"""

class DashcamConfig:
    APP_NAME = "camp_dashcam"
    APP_VERSION = "v1.3.0"

    # 相机安装朝向与 HUD 模式: "auto" (自动由 IMU 识别), "upright" (强制正装), "inverted" (强制倒装)
    MOUNT_MODE = "auto"

    # 真实道路表面与车道边缘检测 (Real Road Perception)
    ROAD_DETECTION_ENABLED = True
    ROAD_CONFIDENCE_THRESHOLD = 0.35
    ROAD_SAMPLE_STEPS = 7

    # 距离平滑与滤波 (EMA Filter: 抑制路面颠簸引起的瞬时微分噪声)
    DISTANCE_SMOOTH_ALPHA = 0.5 # 0.5 平衡实时响应与抗颠簸滤波

    # 虚拟车道走廊过滤 (Virtual Lane Corridor: 过滤邻车道与路侧静止车)
    LANE_CORRIDOR_HALF_WIDTH_M = 1.35  # 本车道单侧走廊半宽 (米，标准城市狭窄车道 2.7m~3.0m -> 半宽 1.35m)
    DEBOUNCE_CONFIRM_FRAMES = 3        # 连续确认帧数防抖 (约 100ms 持续确认才正式告警，过滤单帧颠簸毛刺)
    MAX_LATERAL_DRIFT_SPEED_MPS = 1.2  # 横向切出漂移速度抑制阈值 (m/s，过滤转弯或直道擦身而过的路侧静止车)
    CORRIDOR_Z_NEAR_M = 3.5            # 虚拟车道地毯渲染近端距离 (米)
    CORRIDOR_Z_FAR_M = 25.0            # 虚拟车道地毯渲染远端距离 (米)
    ESTIMATED_SPEED_MPS = 10.0         # 转弯曲率默认估算车速 (m/s, 约 36 km/h)
    MAX_CURVATURE_INV_M = 0.06         # 最大转弯曲率限幅 (1/m, 对应最小转弯半径约 16.7 米，防止过度甩尾)

    # 车载级流式自包含视频录制配置 (断电防丢保护)
    RECORD_VIDEO_ENABLED = True
    VIDEO_CONTAINER_EXT = ".avi" # 流式自包含容器扩展名
    VIDEO_FOURCC = "MJPG"        # 流式 Motion-JPEG 独立帧编码 (抗突发断电报废)
    VIDEO_FPS = 30
    SYNC_INTERVAL_FRAMES = 30    # 每 30 帧 (1 秒) 执行一次底层硬件强制刷盘 (os.sync)


    # 相机光学与安装几何参数 (已适配广角大视野镜头)
    CAMERA_HEIGHT_M = 1.3       # 相机离地高度 (米)
    CAMERA_PITCH_DEG = 0.0      # 相机默认俯仰角 (度, 向上为正/水平为0, 实际由 IMU 重力动态自标定)
    CAMERA_FOV_V_DEG = 65.0     # 广角垂直视场角 (度)
    CAMERA_FOV_H_DEG = 85.0     # 广角水平视场角 (度)
    IMG_WIDTH = 640             # AI 推理输入宽度
    IMG_HEIGHT = 480            # AI 推理输入高度

    # 前向主车道 ROI 扇区过滤 (归一化范围 0.0 ~ 1.0)
    ROI_X_MIN_RATIO = 0.05      # 视场更宽，由虚拟车道物理横坐标精确过滤
    ROI_X_MAX_RATIO = 0.95      # 视场更宽，由虚拟车道物理横坐标精确过滤
    ROI_Y_MIN_RATIO = 0.35      # 忽略过高区域 (天空/远处地平线上方)


    # 目标跟踪与生命周期
    TRACK_MAX_LOST_FRAMES = 5   # 连续丢失 5 帧则自动注销 Track ID
    MIN_DT_SEC = 0.03           # 最小时间差 (30ms)，防止除零或噪声抖动
    MIN_APPROACH_SPEED_MPS = 0.5 # 相对逼近速度阈值 (m/s)，低于此速度视为相对静止或远离

    # FCW 碰撞预警阈值 (TTC: Time-To-Collision)
    MAX_EVAL_DISTANCE_M = 40.0   # 超过 40 米不触发紧急逼近报警
    WARNING_DISTANCE_M = 25.0    # 距离阈值 (米)
    WARNING_TTC_SEC = 2.8        # 黄色注意告警 TTC 阈值 (秒)
    CRITICAL_TTC_SEC = 1.8       # 红色紧急告警 TTC 阈值 (秒)

    # 识别目标类型过滤 (COCO 类别 ID)
    TARGET_PERSON = 0
    TARGET_BICYCLE = 1
    TARGET_CAR = 2
    TARGET_MOTORCYCLE = 3
    TARGET_BUS = 5
    TARGET_TRUCK = 7
    VALID_CLASSES = {TARGET_PERSON, TARGET_BICYCLE, TARGET_CAR, TARGET_MOTORCYCLE, TARGET_BUS, TARGET_TRUCK}

    # G-Sensor (IMU) 碰撞与急刹判据
    IMU_TOTAL_SHOCK_THRESHOLD_G = 1.7  # 总加速度冲击阈值 (G)
    IMU_BRAKE_DECEL_THRESHOLD = 6.0    # 纵向急刹减速度阈值 (m/s²)
    IMU_LOCK_COOLDOWN_SEC = 10.0       # 加锁触发后冷却期 (秒)，防止同一震动连续触发

    # 存储与视频分段管理
    DEFAULT_STORAGE_BASE = "/maixapp/data/dashcam"
    SEGMENT_DURATION_SEC = 180         # 每段视频录制时长 (3 分钟)
    MAX_NORMAL_FILES = 100             # 默认最多保留 100 个分段 (约 5 小时循环录像)


