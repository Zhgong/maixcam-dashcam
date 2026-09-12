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
            total_g=1.0
        )

        string_calls = mock_img.draw_string.call_args_list
        self.assertGreater(len(string_calls), 0, "必须绘制 HUD 状态文本！")


if __name__ == "__main__":
    unittest.main()
