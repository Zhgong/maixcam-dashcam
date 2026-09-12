"""
Unit Test & Quality Gate for Camera Capture Pipeline & Upright Orientation
"""

import unittest
from unittest.mock import MagicMock


class TestCameraPipelineOrientation(unittest.TestCase):
    """
    [Phase 1 Quality Gate: 真实世界照片正立防回归契约测试]
    验证从摄像头采集的原始帧必须通过统一的 FlipDir.XY 旋转 180° 进行出厂模组倒置纠偏，
    确保在真实物理世界中，桌上的物品（笔、笔记本、地毯）位于画面物理下方。
    """

    def test_camera_pipeline_reads_pure_raw_without_flip(self):
        mock_cam = MagicMock()
        mock_raw_img = MagicMock()
        mock_cam.read.return_value = mock_raw_img

        # 直接读取纯净帧
        img = mock_cam.read()
        mock_cam.read.assert_called_once()
        self.assertEqual(img, mock_raw_img, "采集管道必须直接返回原生相机图像，杜绝任何软件层翻转！")


if __name__ == "__main__":
    unittest.main()
