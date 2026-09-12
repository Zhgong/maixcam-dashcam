"""
G-Sensor (IMU) Collision & Hard Braking Detection Module
"""

import math
try:
    from config import DashcamConfig
except ImportError:
    from .config import DashcamConfig



class ShockEventLevel:
    NONE = "NONE"
    BRAKING = "BRAKING"
    CRITICAL_IMPACT = "CRITICAL_IMPACT"


class GSensorGuard:
    def __init__(self, config=DashcamConfig):
        self.config = config
        self.is_locked = False
        self.last_lock_time = 0.0
        self.last_event = None

    def evaluate_imu(self, imu_data: list, now_sec: float) -> dict:
        """
        评估 IMU 传感器读数 (纯数值计算，与底层硬件库解耦)。
        :param imu_data: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, temp] (m/s² 与 dps)
        :param now_sec: 当前时间戳 (秒)
        :return: event dict 或 None
        """
        if not imu_data or len(imu_data) < 3:
            return None

        acc_x, acc_y, acc_z = imu_data[0], imu_data[1], imu_data[2]
        total_acc = math.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        total_g = total_acc / 9.80665
        dyn_g = abs(total_g - 1.0)  # 动态加速度偏差 (消除地球静态 1G 重力影响)

        # 冷却期检查：如果在加锁冷却期内，不重复产生加锁事件
        if now_sec - self.last_lock_time < self.config.IMU_LOCK_COOLDOWN_SEC:
            return None

        event_level = ShockEventLevel.NONE

        # 1. 剧烈碰撞检测 (总冲击 >= 1.6G 或 动态偏差 >= 0.55G)
        if total_g >= self.config.IMU_TOTAL_SHOCK_THRESHOLD_G or dyn_g >= 0.55:
            event_level = ShockEventLevel.CRITICAL_IMPACT
        # 2. 动态急刹/急减速检测 (水平加速度 >= 0.6G 导致向量模长偏差 dyn_g >= 0.20G)
        elif dyn_g >= 0.20:
            event_level = ShockEventLevel.BRAKING


        if event_level != ShockEventLevel.NONE:
            self.is_locked = True
            self.last_lock_time = now_sec
            event = {
                "level": event_level,
                "total_g": float(total_g),
                "dyn_g": float(dyn_g),
                "acc_y": float(acc_y),
                "timestamp": now_sec
            }
            self.last_event = event
            return event

        return None


    def reset_lock(self):
        """手动或录像换段后重置锁定状态"""
        self.is_locked = False
