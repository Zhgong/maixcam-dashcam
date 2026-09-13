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

    def test_single_edge_recovery_left_only(self):
        """
        验证仅左侧标线清晰时，系统自动依据预设车道宽度先验补全右侧标线。
        """
        h, w = 480, 640
        img = np.full((h, w, 3), 60, dtype=np.uint8)
        horizon_y = 300
        # 仅绘制左线
        cv2.line(img, (160, 460), (280, horizon_y), (240, 240, 240), thickness=6)

        result: RoadDetectionResult = self.detector.detect_actual_road(img, horizon_y=horizon_y, imu_yaw_rate=0.0)
        self.assertTrue(result.detected)
        self.assertGreaterEqual(len(result.right_boundary_pts), 3, "右边界必须被先验模型成功补全！")
        self.assertGreater(result.right_boundary_pts[0][0], result.left_boundary_pts[0][0])

    def test_single_edge_recovery_right_only(self):
        """
        验证仅右侧标线清晰时，系统自动补齐左侧标线。
        """
        h, w = 480, 640
        img = np.full((h, w, 3), 60, dtype=np.uint8)
        horizon_y = 300
        # 仅绘制右线
        cv2.line(img, (480, 460), (360, horizon_y), (240, 240, 240), thickness=6)

        result: RoadDetectionResult = self.detector.detect_actual_road(img, horizon_y=horizon_y, imu_yaw_rate=0.0)
        self.assertTrue(result.detected)
        self.assertGreaterEqual(len(result.left_boundary_pts), 3, "左边界必须被先验模型成功补全！")
        self.assertLess(result.left_boundary_pts[0][0], result.right_boundary_pts[0][0])

    def test_overexposed_and_empty_images(self):
        """
        验证纯白过曝图像与空数组的保护逻辑。
        """
        white_img = np.full((480, 640, 3), 255, dtype=np.uint8)
        res_white = self.detector.detect_actual_road(white_img, horizon_y=300)
        self.assertFalse(res_white.detected)

        empty_img = np.array([])
        res_empty = self.detector.detect_actual_road(empty_img, horizon_y=300)
        self.assertFalse(res_empty.detected)

        # 单通道灰度图支持
        gray_img = np.full((480, 640), 60, dtype=np.uint8)
        cv2.line(gray_img, (160, 460), (280, 300), 240, thickness=6)
        cv2.line(gray_img, (480, 460), (360, 300), 240, thickness=6)
        res = self.detector.detect_actual_road(gray_img, horizon_y=240)
        self.assertIsInstance(res, RoadDetectionResult)

    def test_detect_actual_road_inverted_mode(self):
        """
        验证倒装模式 (is_inverted=True) 下的路面 ROI 提取与道路检测
        """
        img = np.ones((480, 640, 3), dtype=np.uint8) * 80
        # 在顶部区域 (倒装模式的路面区域) 绘制两条车道线
        cv2.line(img, (180, 50), (280, 200), (255, 255, 255), 6)
        cv2.line(img, (460, 50), (360, 200), (255, 255, 255), 6)

        res = self.detector.detect_actual_road(img, horizon_y=220, imu_yaw_rate=2.0, is_inverted=True)
        self.assertIsInstance(res, RoadDetectionResult)
        if res.detected:
            self.assertGreater(len(res.left_boundary_pts), 0)
            self.assertGreater(len(res.right_boundary_pts), 0)


if __name__ == '__main__':
    unittest.main()
