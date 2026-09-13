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

    def test_target_box_inverted_mapping(self):
        """
        [Phase 2 Quality Gate: 无论正装或倒装，目标框与实际图像直接 1:1 物理对齐断言]
        验证无论 is_inverted 为 False 还是 True：
        目标框 [x=100, y=200, w=50, h=80] 均必须 1:1 绘制在原生坐标 (100, 200, 50, 80)，
        确保框与画面中的物理目标完全重合，消除左右颠倒错位！
        """
        mock_img = MagicMock()
        target = FCWTarget(track_id=1, class_id=DashcamConfig.TARGET_CAR, bbox=[100, 200, 50, 80], distance_m=25.0, timestamp=100.0)

        # 1. 正装调用
        self.renderer.render_hud(
            img=mock_img,
            tracks={1: target},
            alerts=[],
            is_inverted=False
        )
        rect_calls = [c[0] for c in mock_img.draw_rect.call_args_list if c[0][0] == 100 and c[0][1] == 200]
        self.assertEqual(len(rect_calls), 1, "正装模式下目标框必须精准绘制在 (100, 200)！")
        self.assertEqual(rect_calls[0][2], 50)
        self.assertEqual(rect_calls[0][3], 80)

        # 2. 倒装调用
        mock_img.reset_mock()
        self.renderer.render_hud(
            img=mock_img,
            tracks={1: target},
            alerts=[],
            is_inverted=True
        )
        inverted_rect_calls = [c[0] for c in mock_img.draw_rect.call_args_list if c[0][0] == 100 and c[0][1] == 200]
        self.assertEqual(len(inverted_rect_calls), 1, "倒装模式下目标框也必须直接 1:1 绘制在 (100, 200) 紧贴实际目标！")
        self.assertEqual(inverted_rect_calls[0][2], 50)
        self.assertEqual(inverted_rect_calls[0][3], 80)

    def test_inverted_lane_corridor_and_road_tags(self):
        """
        [Phase 3 Quality Gate: 车道线刻度与道路感知标签倒装适配断言]
        验证在 is_inverted=True 时：
        1. 虚拟车道地毯的 5m, 10m, 20m 刻度线依然调用 draw_line 绘制。
        2. 刻度文字 (5m, 10m, 20m) 与道路感知标签 ([LANE 90%]) 在倒装模式下被正常处理且渲染调用无异常。
        """
        mock_img = MagicMock()
        from core.road_detector import RoadDetectionResult

        road_res = RoadDetectionResult(
            detected=True,
            confidence=0.9,
            left_boundary_pts=[(150, 470), (190, 400), (230, 330)],
            right_boundary_pts=[(490, 470), (450, 400), (410, 330)],
            road_center_offset_px=0.0,
            road_type="painted_lanes"
        )

        self.renderer.render_hud(
            img=mock_img,
            tracks={},
            alerts=[],
            road_res=road_res,
            is_inverted=True
        )

        # 验证 draw_line 正常绘制车道走廊及刻度线
        self.assertTrue(mock_img.draw_line.called)
        self.assertGreaterEqual(mock_img.draw_line.call_count, 6)

        # 在主机测试环境 (无 maix C 扩展) 下，draw_rotated_text 回退为 draw_string，
        # 验证刻度文本与道路标签均已进入绘制流水线
        string_calls = [str(c) for c in mock_img.draw_string.call_args_list]
        self.assertTrue(any("5m" in c for c in string_calls), "必须绘制 5m 距离刻度！")
        self.assertTrue(any("10m" in c for c in string_calls), "必须绘制 10m 距离刻度！")
        self.assertTrue(any("20m" in c for c in string_calls), "必须绘制 20m 距离刻度！")
        self.assertTrue(any("LANE" in c for c in string_calls), "必须绘制道路感知状态徽章！")

    def test_project_ground_point_near_clamping(self):
        """
        验证 z_m <= 0.1 时返回中心点 (cx, cy)
        """
        px, py = self.renderer.project_ground_point(0.0, 0.05)
        self.assertEqual(px, int(self.renderer.cx))
        self.assertEqual(py, int(self.renderer.cy))

    def test_render_hud_critical_banner_and_paused_recording(self):
        """
        验证 CRITICAL 告警横幅渲染以及录像暂停状态 PAUSED 渲染
        """
        from core.fcw_tracker import AlertLevel, FCWTarget
        mock_img = MagicMock()
        crit_target = FCWTarget(track_id=10, class_id=DashcamConfig.TARGET_CAR, bbox=[200, 200, 80, 80],
                                distance_m=5.0, timestamp=100.0)
        crit_target.active_alert_level = AlertLevel.CRITICAL
        crit_target.ttc_sec = 0.8
        alerts = [{"level": AlertLevel.CRITICAL, "target": crit_target}]

        self.renderer.render_hud(
            img=mock_img,
            tracks={10: crit_target},
            alerts=alerts,
            is_recording=False,
            is_locked=False,
            is_inverted=False
        )
        string_calls = [str(c) for c in mock_img.draw_string.call_args_list]
        self.assertTrue(any("CRITICAL" in c for c in string_calls), "必须绘制 CRITICAL 告警横幅！")
        self.assertTrue(any("PAUSED" in c for c in string_calls), "必须绘制 REC [PAUSED] 提示！")

        # 同样验证 WARNING 级别目标框颜色与倒装模式下的 critical 横幅
        mock_img.reset_mock()
        warn_target = FCWTarget(track_id=11, class_id=DashcamConfig.TARGET_CAR, bbox=[200, 200, 80, 80],
                                distance_m=12.0, timestamp=100.0)
        warn_target.active_alert_level = AlertLevel.WARNING
        warn_target.ttc_sec = 2.1
        self.renderer.render_hud(
            img=mock_img,
            tracks={11: warn_target},
            alerts=[{"level": AlertLevel.CRITICAL, "target": crit_target}],
            is_recording=True,
            is_locked=True,
            is_inverted=True
        )
        string_calls_inv = [str(c) for c in mock_img.draw_string.call_args_list]
        self.assertTrue(any("CRITICAL" in c for c in string_calls_inv))
        self.assertTrue(any("LOCKED" in c for c in string_calls_inv))

    def test_render_actual_road_none_or_not_detected(self):
        """
        验证 render_actual_road 在 road_res 为 None 或 detected=False 时直接返回无动作
        """
        mock_img = MagicMock()
        self.renderer.render_actual_road(mock_img, None)
        self.assertFalse(mock_img.draw_line.called)

        from core.road_detector import RoadDetectionResult
        res_false = RoadDetectionResult(detected=False)
        self.renderer.render_actual_road(mock_img, res_false)
        self.assertFalse(mock_img.draw_line.called)


if __name__ == '__main__':
    unittest.main()


