"""
Unit Test & Quality Gate for HUD Text Layout (Phase 2)
"""

import unittest
from unittest.mock import MagicMock
from core.config import DashcamConfig
from core.hud_renderer import HUDRenderer


class TestHUDTextLayout(unittest.TestCase):
    """
    [Phase 2 Quality Gate: HUD 文字布局契约测试]
    验证状态栏与文字渲染在标准区域，保证物理屏幕上的自然可读性。
    """

    def setUp(self):
        self.renderer = HUDRenderer(config=DashcamConfig)

    def test_status_bar_text_positioned_properly(self):
        mock_img = MagicMock()
        self.renderer.render_hud(
            img=mock_img,
            tracks={},
            alerts=[],
            is_recording=True,
            is_locked=False,
            rec_seconds=10,
            temp_c=45,
            total_g=1.0,
            is_inverted=False
        )

        string_calls = mock_img.draw_string.call_args_list
        self.assertGreater(len(string_calls), 0, "必须绘制 HUD 状态文本！")
        # 正装模式下：状态顶栏在 y=8，底部 Exit 按钮在 y=460 (480-20)
        top_text_calls = [c[0] for c in string_calls if c[0][1] == 8]
        bottom_text_calls = [c[0] for c in string_calls if c[0][1] == 460]
        self.assertGreaterEqual(len(top_text_calls), 2, "正装模式下顶栏文字必须位于 y=8！")
        self.assertGreaterEqual(len(bottom_text_calls), 1, "正装模式下底栏文字必须位于 y=460！")

    def test_status_bar_inverted_position(self):
        """
        [Phase 1 Quality Gate: 倒装模式下 HUD 顶栏与底栏自动互换位置契约测试]
        验证当 is_inverted=True 时：
        1. 状态顶栏移动至图像底部 y=456 (480-24)，对应驾驶员肉眼所见的物理屏幕上方；
        2. 底部信息条与 Exit 移动至图像顶部 y=6，对应驾驶员肉眼所见的物理屏幕下方。
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
            total_g=1.0,
            is_inverted=True
        )

        string_calls = mock_img.draw_string.call_args_list
        # 倒装模式下：录像状态和温度在 y=456，Exit 按钮在 y=6
        top_text_calls = [c[0] for c in string_calls if c[0][1] == 456]
        bottom_text_calls = [c[0] for c in string_calls if c[0][1] == 6]
        self.assertGreaterEqual(len(top_text_calls), 2, "倒装模式下状态栏文字必须位于 y=456 (物理屏幕顶)！")
        self.assertGreaterEqual(len(bottom_text_calls), 1, "倒装模式下底栏 Exit 必须位于 y=6 (物理屏幕底)！")


if __name__ == "__main__":
    unittest.main()
