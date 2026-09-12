"""
Unit Tests for Local YOLO11 Detector (ONNX Runtime inference & formatting)
"""

import os
import unittest
import numpy as np
from tools.yolo_detector import YOLO11Detector
from core.config import DashcamConfig


class TestYOLO11Detector(unittest.TestCase):

    def setUp(self):
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yolo11n.onnx")
        self.detector = YOLO11Detector(model_path=self.model_path, conf_thresh=0.25, iou_thresh=0.45)

    def test_detector_initialization(self):
        """
        验证 YOLO11 检测器加载与会话建立。
        """
        self.assertTrue(self.detector.is_ready)
        self.assertIsNotNone(self.detector.session)

    def test_inference_on_synthetic_image(self):
        """
        验证在合成图像上的前向推理与输出数据结构契约。
        输出必须为: [{'track_id': int, 'class_id': int, 'bbox': [x, y, w, h], 'score': float}]
        """
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = self.detector.detect(dummy_img)

        self.assertIsInstance(detections, list)
        for det in detections:
            self.assertIn('track_id', det)
            self.assertIn('class_id', det)
            self.assertIn('bbox', det)
            self.assertIn('score', det)
            self.assertEqual(len(det['bbox']), 4)
            # class_id 必须在关注类别白名单中
            self.assertIn(det['class_id'], DashcamConfig.VALID_CLASSES)

    def test_class_filtering(self):
        """
        验证非关注类别 (非车/人/骑行者) 被自动滤除。
        """
        allowed = {
            DashcamConfig.TARGET_PERSON,
            DashcamConfig.TARGET_BICYCLE,
            DashcamConfig.TARGET_CAR,
            DashcamConfig.TARGET_MOTORCYCLE,
            DashcamConfig.TARGET_BUS,
            DashcamConfig.TARGET_TRUCK
        }
        self.assertEqual(allowed, DashcamConfig.VALID_CLASSES)


if __name__ == "__main__":
    unittest.main()
