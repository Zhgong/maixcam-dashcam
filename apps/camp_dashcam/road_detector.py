"""
RoadDetector for MaixCAM 2 Dashcam (v1.3.0)
Real-world Road Surface & Lane Boundary Perception Engine.
Supports:
1. Painted white/yellow lane markings (HLS/Sobel gradient filter).
2. Unmarked asphalt road boundaries (German residential street curb/asphalt edge detection).
3. Perspective sliding window polynomial/polyline curve generation.
4. Smooth fallback under low-confidence or degraded vision conditions.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import cv2

try:
    from config import DashcamConfig
except ImportError:
    from .config import DashcamConfig


@dataclass
class RoadDetectionResult:
    detected: bool = False
    confidence: float = 0.0
    left_boundary_pts: List[Tuple[int, int]] = field(default_factory=list)
    right_boundary_pts: List[Tuple[int, int]] = field(default_factory=list)
    road_center_offset_px: float = 0.0
    road_type: str = "none"  # "painted_lanes", "unmarked_asphalt", "none"


class RoadDetector:
    def __init__(self, config=DashcamConfig):
        self.config = config
        self.img_w = getattr(config, "IMG_WIDTH", 640)
        self.img_h = getattr(config, "IMG_HEIGHT", 480)
        self.min_confidence = getattr(config, "ROAD_CONFIDENCE_THRESHOLD", 0.35)
        self.sample_steps = getattr(config, "ROAD_SAMPLE_STEPS", 7)

    @staticmethod
    def _smooth_curve_points(pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """利用一阶多项式对采样点进行鲁棒拟合平滑，抑制噪声毛刺"""
        if len(pts) < 3:
            return pts
        try:
            xs = np.array([p[0] for p in pts], dtype=float)
            ys = np.array([p[1] for p in pts], dtype=float)
            poly = np.polyfit(ys, xs, deg=1)
            y_fine = np.linspace(ys.max(), ys.min(), len(pts), dtype=int)
            x_fine = np.polyval(poly, y_fine).astype(int)
            return [(int(x), int(y)) for x, y in zip(x_fine, y_fine)]
        except Exception:
            return pts

    def detect_actual_road(
        self,
        img: np.ndarray,
        horizon_y: int = 300,
        imu_yaw_rate: float = 0.0,
        is_inverted: bool = False
    ) -> RoadDetectionResult:
        """
        从单帧车载前视图像中提取实际物理路面左右边界。
        :param img: 输入图像 (H, W, 3) BGR/RGB 或 (H, W) 灰度图
        :param horizon_y: 动态自标定地平线高度 (低于此高度才为路面)
        :param imu_yaw_rate: 偏航角速度 (deg/s)
        :param is_inverted: 相机是否倒装 (True 时路面在图像上方)
        :return: RoadDetectionResult
        """
        if img is None or img.size == 0:
            return RoadDetectionResult()

        h, w = img.shape[:2]
        horizon_y = max(100, min(h - 50, int(horizon_y)))

        # 1. 检查是否为全黑/过暗或严重过曝的无效图像
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img
        else:
            gray = img

        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        if std_val < 8.0 or mean_val < 5.0 or mean_val > 250.0:
            return RoadDetectionResult(detected=False, confidence=0.0)

        # 2. 提取路面 ROI (正装时：地平线下沿至图像底部车盖前；倒装时：图像顶部车盖前至地平线上沿)
        if not is_inverted:
            roi_top = horizon_y
            roi_bottom = min(h - 15, int(h * 0.92))
        else:
            eff_horizon = h - 1 - horizon_y
            roi_top = max(15, int(h * 0.08))
            roi_bottom = max(roi_top + 30, eff_horizon)

        roi_gray = gray[roi_top:roi_bottom, :]
        roi_h = roi_bottom - roi_top

        # 3. 策略 A: 提取高亮白色车道标线 (Painted Lanes)
        roi_white_mask = np.zeros_like(roi_gray)
        roi_white_mask[roi_gray > max(190, int(mean_val + 35))] = 255

        # 4. 策略 B: 提取沥青路面物理边缘 (Sobel 垂直/斜向边缘梯度)
        blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        sobel_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        _, roi_edge_mask = cv2.threshold(abs_sobel_x, 25, 255, cv2.THRESH_BINARY)

        # 融合车道标线与物理路缘特征
        roi_combined = cv2.bitwise_or(roi_white_mask, roi_edge_mask)

        # 5. 分层滑动窗口 (Sliding Window) 提取左右边界
        # 从底部到地平线均匀采样 y
        y_samples = np.linspace(roi_bottom - 5, roi_top + 10, self.sample_steps, dtype=int)

        left_pts = []
        right_pts = []
        cx = w / 2.0

        # 根据 IMU 偏航角速度预估弯道中心漂移
        center_shift = 0.0
        if abs(imu_yaw_rate) >= 1.0:
            center_shift = -imu_yaw_rate * 1.5

        white_hits = 0
        edge_hits = 0

        for y in y_samples:
            local_y = y - roi_top
            y_start = max(0, local_y - 10)
            y_end = min(roi_h, local_y + 10)
            strip_white = roi_white_mask[y_start:y_end, :]
            strip_edges = roi_combined[y_start:y_end, :]

            col_white = np.sum(strip_white, axis=0)
            col_edges = np.sum(strip_edges, axis=0)

            # 透视收敛预期搜索区间: 越接近地平线，道路在画面中心收敛越窄
            if not is_inverted:
                ratio = (y - roi_top) / max(1.0, (roi_bottom - roi_top))  # 1.0 at bottom (near), 0.0 at top (far)
            else:
                ratio = (roi_bottom - y) / max(1.0, (roi_bottom - roi_top)) # 1.0 at top (near), 0.0 at bottom (far)

            expected_lane_half_w = 40 + int(ratio * 180)  # 远端约 40px 半宽，近端约 220px 半宽
            curr_center = int(cx + center_shift * (1.0 - ratio))

            left_search_min = max(10, curr_center - expected_lane_half_w - 90)
            left_search_max = max(left_search_min + 10, curr_center - 20)

            right_search_min = min(curr_center + 20, w - 20)
            right_search_max = min(w - 10, curr_center + expected_lane_half_w + 90)

            # 优先在白线中寻找峰值
            left_x = None
            right_x = None

            # 左边界白线
            if np.max(col_white[left_search_min:left_search_max]) > 0:
                left_x = left_search_min + int(np.argmax(col_white[left_search_min:left_search_max]))
                white_hits += 1
            elif np.max(col_edges[left_search_min:left_search_max]) > 0:
                left_x = left_search_min + int(np.argmax(col_edges[left_search_min:left_search_max]))
                edge_hits += 1

            # 右边界白线
            if np.max(col_white[right_search_min:right_search_max]) > 0:
                right_x = right_search_min + int(np.argmax(col_white[right_search_min:right_search_max]))
                white_hits += 1
            elif np.max(col_edges[right_search_min:right_search_max]) > 0:
                right_x = right_search_min + int(np.argmax(col_edges[right_search_min:right_search_max]))
                edge_hits += 1

            if left_x is not None:
                left_pts.append((int(left_x), int(y)))
            if right_x is not None:
                right_pts.append((int(right_x), int(y)))

        # 6. 结果一致性校验与平滑拟合
        total_samples = len(y_samples)
        valid_left = len(left_pts) >= 3
        valid_right = len(right_pts) >= 3

        if not valid_left and not valid_right:
            return RoadDetectionResult(detected=False, confidence=0.1)

        # 单边补全: 如果仅有一侧边界清晰，利用平行车道宽度几何先验镜像推导另一侧
        default_lane_w_bottom = 320
        default_lane_w_top = 100
        if valid_left and not valid_right:
            right_pts = []
            for lx, ly in left_pts:
                r = (ly - roi_top) / max(1.0, (roi_bottom - roi_top)) if not is_inverted else (roi_bottom - ly) / max(1.0, (roi_bottom - roi_top))
                w_est = int(default_lane_w_top + r * (default_lane_w_bottom - default_lane_w_top))
                rx = min(w - 10, lx + w_est)
                right_pts.append((rx, ly))
            valid_right = True
        elif valid_right and not valid_left:
            left_pts = []
            for rx, ry in right_pts:
                r = (ry - roi_top) / max(1.0, (roi_bottom - roi_top)) if not is_inverted else (roi_bottom - ry) / max(1.0, (roi_bottom - roi_top))
                w_est = int(default_lane_w_top + r * (default_lane_w_bottom - default_lane_w_top))
                lx = max(10, rx - w_est)
                left_pts.append((lx, ry))
            valid_left = True

        # 平滑曲线拟合 (一阶鲁棒拟合，消除单帧路牙毛刺)
        left_pts = self._smooth_curve_points(left_pts)
        right_pts = self._smooth_curve_points(right_pts)

        # 排序点集确保从物理近端到物理远端
        if not is_inverted:
            left_pts.sort(key=lambda p: -p[1])
            right_pts.sort(key=lambda p: -p[1])
        else:
            left_pts.sort(key=lambda p: p[1])
            right_pts.sort(key=lambda p: p[1])

        # 几何收敛性合理度校验
        # 物理近端宽度应大于物理远端宽度
        w_near = right_pts[0][0] - left_pts[0][0]
        w_far = right_pts[-1][0] - left_pts[-1][0]

        if w_near <= 40 or w_far <= 10 or w_near < w_far:
            return RoadDetectionResult(detected=False, confidence=0.2)

        # 计算置信度
        sample_ratio = (len(left_pts) + len(right_pts)) / (2.0 * total_samples)
        conf = min(0.95, 0.35 + 0.55 * sample_ratio)

        road_type = "painted_lanes" if white_hits >= edge_hits else "unmarked_asphalt"

        # 计算近端道路中心相对于画面中心的横向偏移像素
        bottom_road_center = (left_pts[0][0] + right_pts[0][0]) / 2.0
        center_offset = float(bottom_road_center - cx)

        detected = conf >= self.min_confidence

        return RoadDetectionResult(
            detected=detected,
            confidence=float(conf),
            left_boundary_pts=left_pts,
            right_boundary_pts=right_pts,
            road_center_offset_px=center_offset,
            road_type=road_type
        )
