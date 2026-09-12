"""
Unit test & Quality Gate for Version Management & Architecture Consistency
"""

import os
import unittest
from unittest.mock import MagicMock
import yaml
from core.config import DashcamConfig
from core.hud_renderer import HUDRenderer

APP_DIR = os.path.join(os.path.dirname(__file__), "../apps/camp_dashcam")
APP_YAML = os.path.join(APP_DIR, "app.yaml")


class TestVersionAndArchitecture(unittest.TestCase):

    def test_version_alignment_between_config_and_app_yaml(self):
        """
        [Version Gate]: 确保 config.py 中的 APP_VERSION (如 v1.0.1)
        与 app.yaml 中的 version (如 1.0.1) 严格一致！
        """
        self.assertTrue(os.path.exists(APP_YAML), "app.yaml 必须存在于 app 目录中！")
        with open(APP_YAML, "r", encoding="utf-8") as f:
            app_meta = yaml.safe_load(f)

        yaml_version = str(app_meta.get("version", ""))
        config_version = DashcamConfig.APP_VERSION.lstrip("v")

        self.assertEqual(
            yaml_version,
            config_version,
            f"版本不一致！config.py 是 {DashcamConfig.APP_VERSION}，但 app.yaml 是 {yaml_version}"
        )

    def test_hud_displays_version_badge(self):
        """
        [HUD Gate]: 确保 HUD 渲染器在界面上绘制了当前版本信息。
        """
        renderer = HUDRenderer(config=DashcamConfig)
        mock_img = MagicMock()

        renderer.render_hud(
            img=mock_img,
            tracks={},
            alerts=[],
            is_recording=True,
            is_locked=False,
            rec_seconds=10,
            temp_c=45,
            total_g=1.0
        )

        # 检查 draw_string 调用中是否包含版本字符串
        drawn_strings = [call.args[2] for call in mock_img.draw_string.call_args_list if len(call.args) >= 3]
        has_version = any(DashcamConfig.APP_VERSION in s for s in drawn_strings)

        self.assertTrue(has_version, f"HUD 渲染的文字列表中未找到版本号 {DashcamConfig.APP_VERSION}！已绘制: {drawn_strings}")


if __name__ == "__main__":
    unittest.main()
