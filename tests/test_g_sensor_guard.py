"""
Unit tests for G-Sensor Collision Guard (IMU Deceleration & Shock Detection)
"""

import unittest
from core.config import DashcamConfig
from core.g_sensor_guard import GSensorGuard, ShockEventLevel


class TestGSensorGuard(unittest.TestCase):

    def setUp(self):
        self.config = DashcamConfig
        self.guard = GSensorGuard(config=self.config)

    def test_normal_driving_vibration_no_lock(self):
        """
        正常驾驶工况：常规路面颠簸 (总加速度 ~1.0G，平稳行驶)，不应触发加锁。
        """
        # 静止或轻微颠簸: 1G 重力 (z=9.8, x=0.5, y=0.5)
        # total_g = sqrt(9.8^2 + 0.5^2 + 0.5^2) / 9.80665 ≈ 1.004 G
        imu_data = [0.5, 0.5, 9.8, 0.0, 0.0, 0.0, 45.0]
        event = self.guard.evaluate_imu(imu_data, now_sec=100.0)

        self.assertIsNone(event)
        self.assertFalse(self.guard.is_locked)

    def test_static_gravity_on_any_axis_no_false_alarm(self):
        """
        测试静态重力在任意轴上 (如相机竖放时 Y 轴为 9.8m/s²) 绝对不应触发急刹或碰撞误报！
        """
        # 1. 竖立安装 (Y 轴承受 1G 地球重力 9.8 m/s²)
        imu_upright = [0.0, 9.80665, 0.0, 0.0, 0.0, 0.0, 45.0]
        event_upright = self.guard.evaluate_imu(imu_upright, now_sec=100.0)
        self.assertIsNone(event_upright, "竖立静止安装时 Y 轴 9.8m/s² 不应误触发急刹！")

        # 2. 侧立安装 (X 轴承受 1G 重力 9.8 m/s²)
        imu_sideways = [9.80665, 0.0, 0.0, 0.0, 0.0, 0.0, 45.0]
        event_side = self.guard.evaluate_imu(imu_sideways, now_sec=100.0)
        self.assertIsNone(event_side, "侧立静止安装时 X 轴 9.8m/s² 不应误触发急刹！")


    def test_hard_braking_triggers_lock(self):
        """
        急刹车工况：纵向减速度强烈 (例如 y 轴减速度达到 6.5 m/s²)，触发 BRAKING 加锁。
        """
        # 紧急制动: y 轴产生强减速度 7.0 m/s², z=9.8
        imu_data = [0.2, 7.0, 9.8, 0.0, 0.0, 0.0, 45.0]
        event = self.guard.evaluate_imu(imu_data, now_sec=100.0)

        self.assertIsNotNone(event)
        self.assertEqual(event['level'], ShockEventLevel.BRAKING)
        self.assertTrue(self.guard.is_locked)

    def test_severe_impact_triggers_critical_lock(self):
        """
        强烈碰撞工况：总加速度达到 1.8G 以上，立即触发 CRITICAL 碰撞加锁。
        """
        # 剧烈撞击: x=12.0, y=10.0, z=9.8 -> total = sqrt(144+100+96) = 18.4 m/s² ≈ 1.88 G
        imu_data = [12.0, 10.0, 9.8, 0.0, 0.0, 0.0, 45.0]
        event = self.guard.evaluate_imu(imu_data, now_sec=100.0)

        self.assertIsNotNone(event)
        self.assertEqual(event['level'], ShockEventLevel.CRITICAL_IMPACT)
        self.assertTrue(self.guard.is_locked)

    def test_lock_cooldown_prevents_duplicate_spam(self):
        """
        冷却抑制机制：在一次碰撞加锁后，在 COOLDOWN_SEC 冷却期内再次收到震动，不应重复抛出新的加锁事件。
        """
        t0 = 100.0
        severe_imu = [12.0, 10.0, 9.8, 0.0, 0.0, 0.0, 45.0]
        event1 = self.guard.evaluate_imu(severe_imu, now_sec=t0)
        self.assertIsNotNone(event1)

        # 2 秒后（冷却期内）再次收到震动
        t1 = 102.0
        event2 = self.guard.evaluate_imu(severe_imu, now_sec=t1)
        self.assertIsNone(event2, "冷却期内不应重复抛出重复加锁事件！")

        # 超过冷却期 (例如 15 秒后)
        t2 = 120.0
        event3 = self.guard.evaluate_imu(severe_imu, now_sec=t2)
        self.assertIsNotNone(event3, "超出冷却期后若发生二次撞击，应支持重新触发！")

    def test_invalid_data_graceful_handling(self):
        """
        容错保护：IMU 偶发丢包或返回 None 时，优雅处理不崩溃。
        """
        event1 = self.guard.evaluate_imu(None, now_sec=100.0)
        event2 = self.guard.evaluate_imu([], now_sec=100.0)
        event3 = self.guard.evaluate_imu([1.0, 2.0], now_sec=100.0) # 长度不足

        self.assertIsNone(event1)
        self.assertIsNone(event2)
        self.assertIsNone(event3)


if __name__ == '__main__':
    unittest.main()
