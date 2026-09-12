"""
Unit tests for RoadDetector: Real-world Road Surface & Lane Line Extraction (v1.3.0)
"""

import unittest
import numpy as np
import cv2
from core.config import DashcamConfig
from core.road_detector import RoadDetector, RoadDetectionResult


class TestRoadDetector(unittest.TestCase):

    def setUp(self):
        self.config = DashcamConfig
        self.detector = RoadDetector(config=self.config)

    def test_detect_painted_lane_lines(self):
        """
        [Real Road Perception Gate - Painted Lanes]:
        在合成道路图像（深灰沥青地面 + 白色中央虚线 + 左右白色边缘线）上，
        验证检测器能准确识别出左右车道线，并拟合出非空且置信度大于阈值的多项式点集。
        """
        h, w = 480, 640
        # 创建深灰沥青地面背景
        img = np.full((h, w, 3), 60, dtype=np.uint8)

        # 绘制梯形收敛的白色车道线 (左线与右线)
        horizon_y = 300
        # 左线: 底部 (x=160, y=460) -> 顶部 (x=280, y=horizon_y)
        # 右线: 底部 (x=480, y=460) -> 顶部 (x=360, y=horizon_y)
        cv2.line(img, (160, 460), (280, horizon_y), (240, 240, 240), thickness=6)
        cv2.line(img, (480, 460), (360, horizon_y), (240, 240, 240), thickness=6)

        result: RoadDetectionResult = self.detector.detect_actual_road(img, horizon_y=horizon_y, imu_yaw_rate=0.0)

        self.assertTrue(result.detected, "合成明显车道线场景必须检出道路！")
        self.assertGreater(result.confidence, 0.4)
        self.assertIsNotNone(result.left_boundary_pts)
        self.assertIsNotNone(result.right_boundary_pts)
        self.assertGreater(len(result.left_boundary_pts), 3)
        self.assertGreater(len(result.right_boundary_pts), 3)

        # 验证左边界点横坐标在画面左侧，右边界点横坐标在画面右侧
        left_bottom_x = result.left_boundary_pts[0][0]
        right_bottom_x = result.right_boundary_pts[0][0]
        self.assertLess(left_bottom_x, 320)
        self.assertGreater(right_bottom_x, 320)
        self.assertGreater(right_bottom_x, left_bottom_x)

    def test_unmarked_road_edge_extraction(self):
        """
        [Real Road Perception Gate - Unmarked Asphalt Road]:
        在无标线道路（德国居民区：深灰色路面 vs 浅灰色路沿石/人行道）上，
        验证边缘提取器根据沥青与路牙对比度提取实际路面物理轮廓。
        """
        h, w = 480, 640
        # 左侧与右侧为浅色路沿/人行道 (灰度 180)
        img = np.full((h, w, 3), 180, dtype=np.uint8)
        # 中间为深灰色沥青路面 (灰度 50)，梯形延伸
        pts = np.array([[120, 480], [520, 480], [380, 300], [260, 300]], dtype=np.int32)
        cv2.fillPoly(img, [pts], (50, 50, 50))

        result: RoadDetectionResult = self.detector.detect_actual_road(img, horizon_y=300, imu_yaw_rate=0.0)

        self.assertTrue(result.detected, "沥青与路牙对比明显时应识别出道路轮廓！")
        self.assertGreaterEqual(len(result.left_boundary_pts), 3)

    def test_low_confidence_fallback_to_kinematics(self):
        """
        [Fallback Gate]:
        全黑或严重过曝无效画面下，检测器应置 detected=False，平滑回退，不输出错误虚假坐标。
        """
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect_actual_road(blank_img, horizon_y=300, imu_yaw_rate=0.0)
        self.assertFalse(result.detected)
        self.assertLess(result.confidence, 0.3)


if __name__ == '__main__':
    unittest.main()
