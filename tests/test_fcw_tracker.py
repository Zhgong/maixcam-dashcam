"""
Unit tests for FCW Tracker & Monocular Distance / TTC Calculation (v1.2.0 Virtual Lane & Debounce)
"""

import unittest
import math
from core.config import DashcamConfig
from core.fcw_tracker import (
    FCWAnalyzer, FCWTarget, AlertLevel,
    GravityDirection, BottomEdge, OrientationDetector
)


class TestFCWTracker(unittest.TestCase):

    def setUp(self):
        self.config = DashcamConfig
        self.analyzer = FCWAnalyzer(config=self.config)

    def test_geometric_distance_estimation(self):
        """
        验证单目测距几何运算：
        底部接地点越靠下（y_bottom 越大），距离越近；越接近画面中心（地平线），距离越远。
        """
        dist_near = self.analyzer.estimate_distance(bottom_y=480)
        dist_far = self.analyzer.estimate_distance(bottom_y=300)

        self.assertGreater(dist_far, dist_near)
        self.assertGreater(dist_near, 0.5)
        self.assertLess(dist_near, 5.0)

        dist_sky = self.analyzer.estimate_distance(bottom_y=240)
        self.assertEqual(dist_sky, self.config.MAX_EVAL_DISTANCE_M * 2)

    def test_lateral_position_estimation(self):
        """
        [Virtual Lane Gate]: 验证横向物理距离估算 X = (x_center - cx) * Z / fx
        画面中心 x=320 处 X=0.0m；偏左 x<320 为负，偏右 x>320 为正。
        """
        # 正中心车辆，距离 10 米
        x_center = self.analyzer.estimate_lateral_x(bbox_center_x=320, distance_z=10.0)
        self.assertAlmostEqual(x_center, 0.0, places=2)

        # 偏右侧车辆 (x=450)，距离 10 米
        x_right = self.analyzer.estimate_lateral_x(bbox_center_x=450, distance_z=10.0)
        self.assertGreater(x_right, 0.5)

        # 偏左侧车辆 (x=190)，距离 10 米
        x_left = self.analyzer.estimate_lateral_x(bbox_center_x=190, distance_z=10.0)
        self.assertLess(x_left, -0.5)

    def test_adjacent_lane_suppression(self):
        """
        [Virtual Lane Corridor Gate]:
        相邻车道 (如右侧车道或路边停靠车，|X| > 1.75m)，即便快速逼近且 TTC 极小，绝对不触发 CRITICAL 或 WARNING 报警！
        """
        t0 = 100.0
        # 目标 88：位于右侧邻车道 (x=520, w=100 -> center_x=570)，即便靠近，也因超出车道走廊而不报警
        # 在 10 米处，fx ~ 772px，dx = 570 - 320 = 250px -> X = 250 * 10 / 772 = 3.23 米 > 1.75 米
        det_frame1 = [{'track_id': 88, 'class_id': DashcamConfig.TARGET_CAR, 'bbox': [500, 250, 100, 100]}]
        self.analyzer.process_detections(det_frame1, now_sec=t0)

        # 0.2 秒后继续逼近
        det_frame2 = [{'track_id': 88, 'class_id': DashcamConfig.TARGET_CAR, 'bbox': [480, 270, 140, 130]}]
        alerts = self.analyzer.process_detections(det_frame2, now_sec=t0 + 0.2)

        # 连续确认 3 帧也不应报警，因为一直在邻车道走廊外
        for i in range(3):
            alerts = self.analyzer.process_detections(det_frame2, now_sec=t0 + 0.3 + i * 0.1)

        self.assertEqual(len(alerts), 0, "邻车道（走廊外）车辆无论逼近速度多快，都不应触发报警！")
        self.assertFalse(self.analyzer.tracks[88].in_lane_corridor)

    def test_alert_debounce_three_frame_confirmation(self):
        """
        [Debounce Gate]:
        必须连续达到 DEBOUNCE_CONFIRM_FRAMES (3 帧) 预警条件才正式输出告警；
        单帧偶发颠簸引起的瞬时 TTC 尖峰会被彻底消除，不会闪现红框！
        """
        t0 = 100.0
        # 初始前车
        det1 = [{'track_id': 50, 'class_id': DashcamConfig.TARGET_CAR, 'bbox': [260, 200, 120, 100]}]
        self.analyzer.process_detections(det1, now_sec=t0)

        # 帧 2: 突发急刹逼近 (满足 TTC < 1.8s 条件第 1 帧) -> 防抖抑制，暂不报警
        det2 = [{'track_id': 50, 'class_id': DashcamConfig.TARGET_CAR, 'bbox': [240, 230, 160, 140]}]
        alerts_f1 = self.analyzer.process_detections(det2, now_sec=t0 + 0.1)
        self.assertEqual(len(alerts_f1), 0, "第 1 帧达到条件应被防抖抑制，避免单帧频闪！")

        # 帧 3: 连续第 2 帧满足条件 -> 防抖抑制
        alerts_f2 = self.analyzer.process_detections(det2, now_sec=t0 + 0.2)
        self.assertEqual(len(alerts_f2), 0, "第 2 帧达到条件仍应处于确认中！")

        # 帧 4: 连续第 3 帧满足条件 -> 触发正式 CRITICAL 报警！
        alerts_f3 = self.analyzer.process_detections(det2, now_sec=t0 + 0.3)
        self.assertGreater(len(alerts_f3), 0, "连续 3 帧确认后必须正式触发报警！")
        self.assertEqual(alerts_f3[0]['level'], AlertLevel.CRITICAL)

    def test_single_frame_glitch_filtered_by_debounce(self):
        """
        [Anti-Flicker Gate]: 颠簸引起的单帧速度毛刺 (第 1 帧突变，第 2 帧恢复) 0 误报！
        """
        t0 = 100.0
        det_normal = [{'track_id': 7, 'class_id': DashcamConfig.TARGET_CAR, 'bbox': [260, 200, 120, 100]}]
        self.analyzer.process_detections(det_normal, now_sec=t0)

        # 突发 1 帧颠簸抖动
        det_glitch = [{'track_id': 7, 'class_id': DashcamConfig.TARGET_CAR, 'bbox': [250, 250, 140, 120]}]
        alerts_glitch = self.analyzer.process_detections(det_glitch, now_sec=t0 + 0.05)
        self.assertEqual(len(alerts_glitch), 0, "单帧颠簸毛刺必须被完全过滤！")

        # 下一帧恢复正常
        alerts_recover = self.analyzer.process_detections(det_normal, now_sec=t0 + 0.10)
        self.assertEqual(len(alerts_recover), 0)

    def test_red_light_standstill_no_alert(self):
        """
        误报抑制：红灯跟车静止状态下，即便距离很近，相对逼近速度为 0，TTC 为无穷大，绝不报警。
        """
        t0 = 100.0
        det = [{'track_id': 1, 'class_id': DashcamConfig.TARGET_CAR, 'bbox': [200, 250, 200, 150]}]
        self.analyzer.process_detections(det, now_sec=t0)

        alerts = self.analyzer.process_detections(det, now_sec=t0 + 0.5)
        self.assertEqual(len(alerts), 0)
        target = self.analyzer.tracks[1]
        self.assertLess(target.rel_speed_mps, self.config.MIN_APPROACH_SPEED_MPS)
        self.assertEqual(target.ttc_sec, float('inf'))

    def test_stale_track_deregistration(self):
        """
        生命周期管理：目标离开视野后自动从活跃表中删除。
        """
        t = 100.0
        det = [{'track_id': 99, 'class_id': DashcamConfig.TARGET_TRUCK, 'bbox': [200, 200, 100, 100]}]
        self.analyzer.process_detections(det, now_sec=t)
        self.assertIn(99, self.analyzer.tracks)

        for i in range(1, 6):
            self.analyzer.process_detections([], now_sec=t + i * 0.1)

        self.assertNotIn(99, self.analyzer.tracks)

    def test_corridor_width_updated_to_1_35m(self):
        """
        [Tighter Corridor Gate]:
        车道走廊半宽升级至 1.35m (适配欧洲/城市狭窄车道 2.7m~3.0m)。
        对于侧边停靠车辆 (|X| = 1.50m)，在旧版 1.75m 内但在新版 1.35m 外，必须判定为走廊外 (in_lane_corridor=False) 且绝对不触发报警。
        """
        target = FCWTarget(track_id=12, class_id=DashcamConfig.TARGET_CAR, bbox=[100, 200, 100, 100], distance_m=6.0, timestamp=100.0)
        # 更新 lateral_x_m = 1.50m (> 1.35m 且 <= 1.75m)
        target.update(bbox=[100, 200, 100, 100], distance_m=5.0, lateral_x_m=1.50, timestamp=100.2, config=self.config)
        self.assertFalse(target.in_lane_corridor, "X=1.50m 应当超出 1.35m 车道走廊半宽！")
        raw_level = self.analyzer.evaluate_raw_alert_level(target)
        self.assertEqual(raw_level, AlertLevel.NONE, "超出 1.35m 车道走廊的目标不应产生任何预警！")

    def test_lateral_drift_suppression_on_turning_parked_car(self):
        """
        [Lateral Drift / Cut-out Gate]:
        拐弯或直道擦身而过路侧停放车辆时，车辆具有显著的横向平移速度 (|vx| > 1.2 m/s 且 |X| > 0.8m)。
        即便纵向迅速靠近 (TTC < 1.8s 且仍在走廊边沿如 X=1.1m)，也应被横向漂移切出逻辑判定为非碰撞对撞物，抑制报警！
        """
        t0 = 100.0
        # 初始帧: 目标位于右侧 X=0.9m，距离 8m
        target = FCWTarget(track_id=33, class_id=DashcamConfig.TARGET_CAR, bbox=[360, 280, 120, 100], distance_m=8.0, timestamp=t0)
        target.lateral_x_m = 0.9

        # 0.1s 后，因本车转弯或绕行，目标纵向逼近到 6.5m (v_rel = 15m/s, TTC = 0.43s),
        # 但横向快速平移至 X=1.15m (dx = 0.25m, vx = 2.5 m/s > 1.2 m/s)
        target.update(bbox=[380, 310, 140, 120], distance_m=6.5, lateral_x_m=1.15, timestamp=t0 + 0.1, config=self.config)

        self.assertGreater(abs(target.lateral_v_mps), 1.2)
        raw_level = self.analyzer.evaluate_raw_alert_level(target)
        self.assertEqual(raw_level, AlertLevel.NONE, "横向快速横移漂移的目标应被切出过滤器抑制报警！")

    def test_curved_corridor_evaluation_in_turn(self):
        """
        [Curved Corridor Gate]:
        在车辆弯道行驶时 (yaw_rate_dps = 25.0, 向左急转)，车道走廊在远端向左偏移。
        在 15 米处，原本正前方 (X = 0.0m) 的物体因车辆转向，已不再处于车道中心，
        而位于左侧弯道上的目标 (如 X = -2.0m) 正好落入弯曲走廊内部！
        """
        offset_straight = self.analyzer.compute_corridor_center_offset(distance_z=15.0, yaw_rate_dps=0.0)
        self.assertAlmostEqual(offset_straight, 0.0, places=2)

        offset_left_turn = self.analyzer.compute_corridor_center_offset(distance_z=15.0, yaw_rate_dps=25.0)
        self.assertLess(offset_left_turn, -1.0, "左转弯时 15 米处走廊中心应向左偏移超过 1.0 米！")

        # 验证在弯道曲率下的走廊判定
        # 在 10 米处，走廊中心向左偏移约 -2.18 米，因此 X = -2.0 米正好落入弯道走廊内
        in_curve = self.analyzer.is_in_corridor(lateral_x=-2.0, distance_z=10.0, yaw_rate_dps=25.0)
        self.assertTrue(in_curve, "左侧弯道上的目标 (X=-2.0m, Z=10m) 应当被判定在弯曲走廊内！")

        straight_car_in_turn = self.analyzer.is_in_corridor(lateral_x=1.0, distance_z=10.0, yaw_rate_dps=25.0)
        self.assertFalse(straight_car_in_turn, "直对原本右侧/正前方的物体在左转时应被判定在弯曲走廊外！")

    def test_imu_inverted_mount_detection(self):
        """
        [Auto Mount Orientation Gate]:
        验证依据 IMU 静态重力分量 acc_y 自动判断相机安装方向：
        - acc_y > 3.0 m/s^2 -> 正装 (is_inverted = False)
        - acc_y < -3.0 m/s^2 -> 倒装 (is_inverted = True)
        """
        self.assertFalse(self.analyzer.is_camera_inverted(acc_y=9.8))
        self.assertTrue(self.analyzer.is_camera_inverted(acc_y=-9.2))

    def test_imu_pitch_auto_leveling_and_horizon_shift(self):
        """
        [Auto Pitch & Horizon Gate]:
        验证当相机存在安装仰角时 (例如吸盘平视略微仰头，acc_z = -3.2m/s^2, acc_y = 9.2m/s^2)：
        - 自动求解出仰角 pitch_deg 约为 -18° ~ -19°；
        - 地平线纵坐标 y_horizon 从假想的画面中心 240px 动态下移至真实路面位置约 290px ~ 310px；
        """
        # 水平平视 (acc_z = 0)
        pitch_level = self.analyzer.compute_pitch_from_gravity(acc_x=0.0, acc_y=9.8, acc_z=0.0)
        self.assertAlmostEqual(pitch_level, 0.0, places=1)
        h_level = self.analyzer.compute_horizon_y(pitch_deg=pitch_level)
        self.assertAlmostEqual(h_level, 240.0, delta=2.0)

        # 仰头安装 (acc_z = -3.2m/s^2)
        pitch_up = self.analyzer.compute_pitch_from_gravity(acc_x=0.0, acc_y=9.2, acc_z=-3.2)
        self.assertLess(pitch_up, -10.0, "仰头安装 pitch 角度应为显著负角！")
        h_up = self.analyzer.compute_horizon_y(pitch_deg=pitch_up)
        self.assertGreater(h_up, 280.0, "仰头安装时，画面上的真实地平线应当明显下移 (> 280px)！")

    def test_slope_and_braking_pitch_immunity(self):
        """
        [Slope & Brake Immunity Gate]:
        验证车辆进入 10% 陡坡或遇到急加减速时：
        利用低通滤波标定固定的安装机械角，地平线不会随瞬态重力跳变！
        """
        # 初始安装角标定 (仰头约 -19°)
        initial_pitch = self.analyzer.update_pitch_with_gravity(acc_x=0.0, acc_y=9.2, acc_z=-3.2)
        self.assertAlmostEqual(initial_pitch, -19.17, places=1)

        # 模拟车辆驶上 10% 坡道 (重力分量在 Z 轴偏移 1.0 m/s^2)
        # 单帧瞬态冲击下，滤波后的 pitch_deg 变化应极小 (< 0.5度)
        smoothed_pitch = self.analyzer.update_pitch_with_gravity(acc_x=0.0, acc_y=9.2, acc_z=-4.2)
        self.assertAlmostEqual(smoothed_pitch, initial_pitch, delta=0.5, msg="坡道瞬态重力倾斜不应导致机械安装角剧烈偏移！")

    def test_orientation_detector_gravity_and_bottom(self):
        """
        [Gravity Direction & Bottom Determination Gate]:
        验证 OrientationDetector:
        1. 启动零值过滤：[0, 0, 0] 初始采样不改变默认状态；
        2. 正装状态：acc_y = +9.8 -> 重力方向 DOWN，画面物理底部对应图像底部 IMAGE_BOTTOM (is_inverted=False)；
        3. 倒装翻转：acc_y 从 +9.8 变为 -9.8 -> 自动切换为 重力方向 UP，画面物理底部对应图像顶部 IMAGE_TOP (is_inverted=True)；
        4. 迟滞死区防抖：在 [-2.5, +2.5] 区间过渡时，保持当前状态不发生高频闪烁。
        """
        detector = OrientationDetector(ema_alpha=0.4, threshold=2.5)

        # 1. 启动未就绪零值
        changed = detector.update(0.0, 0.0, 0.0)
        self.assertFalse(changed)
        self.assertFalse(detector.is_inverted)
        self.assertEqual(detector.gravity_direction, GravityDirection.DOWN)
        self.assertEqual(detector.bottom_edge, BottomEdge.IMAGE_BOTTOM)

        # 2. 正常直立稳定采样 (acc_y = +9.8)
        detector.update(0.0, 9.8, 0.0)
        self.assertFalse(detector.is_inverted)
        self.assertEqual(detector.gravity_direction, GravityDirection.DOWN)
        self.assertEqual(detector.bottom_edge, BottomEdge.IMAGE_BOTTOM)

        # 3. 翻转至倒吊安装 (acc_y = -9.8)
        # 持续 3 帧以完成平滑并越过 -2.5 阈值
        changed = False
        for _ in range(3):
            c = detector.update(0.0, -9.8, 0.0)
            if c:
                changed = True
        self.assertTrue(changed, "翻转倒吊安装应成功触发朝向切换！")
        self.assertTrue(detector.is_inverted)
        self.assertEqual(detector.gravity_direction, GravityDirection.UP)
        self.assertEqual(detector.bottom_edge, BottomEdge.IMAGE_TOP)

        # 4. 死区过渡 (如晃动到 acc_y = 0.0)
        changed_deadband = detector.update(0.0, 0.0, 9.8)
        self.assertFalse(changed_deadband, "死区过渡期间不应误切换状态！")
        self.assertTrue(detector.is_inverted)
        self.assertEqual(detector.gravity_direction, GravityDirection.UP)
        self.assertEqual(detector.bottom_edge, BottomEdge.IMAGE_TOP)

        # 5. 翻转回正装 (acc_y = +9.8)
        for _ in range(3):
            detector.update(0.0, 9.8, 0.0)
        self.assertFalse(detector.is_inverted)
        self.assertEqual(detector.gravity_direction, GravityDirection.DOWN)
        self.assertEqual(detector.bottom_edge, BottomEdge.IMAGE_BOTTOM)


if __name__ == '__main__':
    unittest.main()




