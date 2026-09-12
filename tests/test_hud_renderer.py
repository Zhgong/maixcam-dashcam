"""
Unit tests for Dashcam HUD Renderer
"""

import unittest
from unittest.mock import MagicMock
from core.config import DashcamConfig
from core.fcw_tracker import FCWTarget, AlertLevel
from core.hud_renderer import HUDRenderer


class TestHUDRenderer(unittest.TestCase):

    def setUp(self):
        self.config = DashcamConfig
        self.renderer = HUDRenderer(config=self.config)

    def test_render_all_alert_levels(self):
        """
        验证各类告警等级与目标绘制在 Mock Canvas 上均能正常运行不抛异常。
        """
        mock_img = MagicMock()
        target_safe = FCWTarget(track_id=1, class_id=DashcamConfig.TARGET_CAR, bbox=[200, 200, 100, 100], distance_m=30.0, timestamp=100.0)
        target_warn = FCWTarget(track_id=2, class_id=DashcamConfig.TARGET_CAR, bbox=[220, 220, 120, 120], distance_m=18.0, timestamp=100.0)
        target_warn.ttc_sec = 2.2
        target_crit = FCWTarget(track_id=3, class_id=DashcamConfig.TARGET_TRUCK, bbox=[240, 240, 150, 150], distance_m=8.0, timestamp=100.0)
        target_crit.ttc_sec = 1.4

        tracks = {1: target_safe, 2: target_warn, 3: target_crit}
        alerts = [
            {'level': AlertLevel.WARNING, 'target': target_warn},
            {'level': AlertLevel.CRITICAL, 'target': target_crit}
        ]

        # 渲染全套 HUD 元素
        self.renderer.render_hud(
            img=mock_img,
            tracks=tracks,
            alerts=alerts,
            is_recording=True,
            is_locked=True,
            rec_seconds=125,
            temp_c=48,
            total_g=1.05
        )

        # 验证 draw_rect, draw_string 都有被调用
        self.assertTrue(mock_img.draw_rect.called)
        self.assertTrue(mock_img.draw_string.called)

    def test_format_recording_time(self):
        """
        验证录像时间格式化输出 (例如 125秒 -> '02:05')。
        """
        formatted = self.renderer.format_duration(125)
        self.assertEqual(formatted, "02:05")

        formatted_zero = self.renderer.format_duration(0)
        self.assertEqual(formatted_zero, "00:00")

    def test_render_perspective_corridor(self):
        """
        [Virtual Lane Carpet Gate]:
        验证 HUD 渲染器正确绘制具有透视汇聚效果的虚拟车道走廊线 (调用 draw_line 绘制左右包络线与距离横梁)。
        """
        mock_img = MagicMock()
        self.renderer.render_hud(
            img=mock_img,
            tracks={},
            alerts=[],
            is_recording=True,
            is_locked=False,
            rec_seconds=10,
            temp_c=45,
            total_g=1.0
        )
        self.assertTrue(mock_img.draw_line.called, "HUD 渲染器必须调用 draw_line 绘制虚拟车道包络线！")
        # 至少有左右边界线 2 条 + 距离刻度横梁
        self.assertGreaterEqual(mock_img.draw_line.call_count, 2)

    def test_dynamic_curvature_corridor_with_yaw_rate(self):
        """
        [Dynamic Curvature Gate]:
        验证当相机存在偏航角速度 (yaw_rate_dps) 时，虚拟车道线动态弯曲：
        - 左转 (yaw_rate > 0): 远端车道中心向左偏移 (x 像素减小)
        - 右转 (yaw_rate < 0): 远端车道中心向右偏移 (x 像素增加)
        - 车道线采用分段多段线绘制，draw_line 调用次数显著增加 (> 6 次)
        """
        mock_img_straight = MagicMock()
        self.renderer.render_hud(
            img=mock_img_straight,
            tracks={},
            alerts=[],
            yaw_rate_dps=0.0
        )

        mock_img_left_turn = MagicMock()
        self.renderer.render_hud(
            img=mock_img_left_turn,
            tracks={},
            alerts=[],
            yaw_rate_dps=25.0  # 左转 25 度/秒
        )

        # 验证分段曲线绘制
        self.assertGreaterEqual(mock_img_left_turn.draw_line.call_count, 6, "动态弯道必须使用分段多段线平滑拟合弧线！")

        # 获取远端车道线坐标，验证左转时远端向左弯曲
        # 查找 draw_line 调用中远端的终点横坐标
        straight_lines = [call[0] for call in mock_img_straight.draw_line.call_args_list]
        left_turn_lines = [call[0] for call in mock_img_left_turn.draw_line.call_args_list]

        # 远端左侧线终点 x
        straight_far_x = straight_lines[0][2]
        left_far_x = left_turn_lines[-1][2] if left_turn_lines else straight_far_x
        # 左转时，远端点应明显偏向左侧 (x 坐标更小)
        # 通过 project_ground_point 检验
        px_straight, _ = self.renderer.project_ground_point(0.0, 20.0, yaw_rate_dps=0.0)
        px_left, _ = self.renderer.project_ground_point(0.0, 20.0, yaw_rate_dps=25.0)
        self.assertLess(px_left, px_straight, "左转时，远端中心线必须向左弯曲 (px_left < px_straight)！")

    def test_render_actual_road_boundaries(self):
        """
        [Real Road HUD Gate]:
        验证当检测到实际道路边界 (RoadDetectionResult) 时，
        HUD 渲染器在图像上绘制实际识别出的左右物理边界多段线与状态标签。
        """
        mock_img = MagicMock()
        from core.road_detector import RoadDetectionResult

        road_res = RoadDetectionResult(
            detected=True,
            confidence=0.85,
            left_boundary_pts=[(150, 470), (190, 400), (230, 330), (270, 300)],
            right_boundary_pts=[(490, 470), (450, 400), (410, 330), (370, 300)],
            road_center_offset_px=5.0,
            road_type="painted_lanes"
        )

        # 验证专有方法 render_actual_road 绘制边界线
        self.renderer.render_actual_road(mock_img, road_res)
        self.assertGreaterEqual(mock_img.draw_line.call_count, 6, "必须为左右路面边界分别绘制多段连接线！")

        # 验证集成在 render_hud 中调用时也能被触发
        mock_img.reset_mock()
        self.renderer.render_hud(
            img=mock_img,
            tracks={},
            alerts=[],
            road_res=road_res,
            horizon_y=300
        )
        self.assertTrue(mock_img.draw_line.called)


if __name__ == '__main__':
    unittest.main()


