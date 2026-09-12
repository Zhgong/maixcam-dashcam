"""
YOLO11 ONNX Runtime Detector & Tracker Adapter for Local PC Offline Replay
Outputs detections in the identical format as MaixPy YOLO11 on board.
"""

import os
import cv2
import numpy as np

try:
    from core.config import DashcamConfig
except ImportError:
    from projects.maixcam_dashcam.core.config import DashcamConfig


class YOLO11Detector:
    def __init__(self, model_path: str = None, conf_thresh: float = 0.25, iou_thresh: float = 0.45):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.session = None
        self.is_ready = False
        self.input_name = None
        self.output_name = None
        self.input_shape = (640, 640)
        self.tracks = {}  # 简易 IOU 跟踪器 {track_id: bbox}
        self.next_track_id = 1

        if model_path is None:
            proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(proj_root, "models", "yolo11n.onnx")

        if os.path.exists(model_path):
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
                self.is_ready = True
            except Exception as e:
                print(f"⚠️ [YOLO11Detector] 加载 ONNX 模型失败: {e}")
        else:
            print(f"⚠️ [YOLO11Detector] 模型文件不存在: {model_path}")

    def preprocess(self, img_bgr: np.ndarray):
        """
        Letterbox 预处理至 640x640，保持纵横比
        """
        h, w = img_bgr.shape[:2]
        target_w, target_h = self.input_shape
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        top = (target_h - nh) // 2
        left = (target_w - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized

        # 归一化并调整通道顺序 [H, W, C] -> [1, C, H, W]
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        meta = {
            "scale": scale,
            "top": top,
            "left": left,
            "orig_w": w,
            "orig_h": h
        }
        return blob, meta

    def postprocess(self, output: np.ndarray, meta: dict):
        """
        解析 YOLO11 输出 [1, 84, 8400]
        前 4 行是 [cx, cy, w, h]，后 80 行是各类别置信度
        """
        # 转置为 [8400, 84]
        predictions = np.transpose(output[0], (1, 0))

        boxes = []
        scores = []
        class_ids = []

        scale = meta["scale"]
        pad_x = meta["left"]
        pad_y = meta["top"]

        valid_classes = DashcamConfig.VALID_CLASSES

        for pred in predictions:
            cls_scores = pred[4:]
            class_id = np.argmax(cls_scores)
            score = float(cls_scores[class_id])

            if score >= self.conf_thresh and class_id in valid_classes:
                cx, cy, w, h = pred[:4]
                # 逆 Letterbox 变换回原图物理像素坐标
                x1 = (cx - w / 2.0 - pad_x) / scale
                y1 = (cy - h / 2.0 - pad_y) / scale
                bw = w / scale
                bh = h / scale

                # 约束边界
                x1 = max(0, min(meta["orig_w"] - 1, x1))
                y1 = max(0, min(meta["orig_h"] - 1, y1))
                bw = max(1, min(meta["orig_w"] - x1, bw))
                bh = max(1, min(meta["orig_h"] - y1, bh))

                boxes.append([int(x1), int(y1), int(bw), int(bh)])
                scores.append(score)
                class_ids.append(int(class_id))

        if not boxes:
            return []

        # NMS 非极大值抑制
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thresh, self.iou_thresh)
        if len(indices) == 0:
            return []

        keep = indices.flatten()
        detections = []
        for i in keep:
            detections.append({
                "bbox": boxes[i],
                "score": scores[i],
                "class_id": class_ids[i]
            })

        return self.assign_track_ids(detections)

    def assign_track_ids(self, detections: list) -> list:
        """
        简易高效的基于 IOU 的目标连续 Track ID 关联器
        """
        matched_tracks = {}
        out_detections = []

        for det in detections:
            bbox = det["bbox"]
            best_iou = 0.0
            best_id = None

            for tid, tbox in self.tracks.items():
                iou = self.compute_iou(bbox, tbox)
                if iou > best_iou and iou >= 0.3:
                    best_iou = iou
                    best_id = tid

            if best_id is not None and best_id not in matched_tracks:
                assigned_id = best_id
            else:
                assigned_id = self.next_track_id
                self.next_track_id += 1

            matched_tracks[assigned_id] = bbox
            det["track_id"] = assigned_id
            out_detections.append(det)

        self.tracks = matched_tracks
        return out_detections

    @staticmethod
    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        denom = float(boxAArea + boxBArea - interArea)
        if denom <= 0:
            return 0.0
        return interArea / denom

    def detect(self, img_bgr: np.ndarray) -> list:
        """
        主检测接口：输入 BGR 画面，返回检测到的目标列表
        :return: [{'track_id': int, 'class_id': int, 'bbox': [x, y, w, h], 'score': float}]
        """
        if not self.is_ready:
            return []

        blob, meta = self.preprocess(img_bgr)
        res = self.session.run([self.output_name], {self.input_name: blob})
        return self.postprocess(res[0], meta)
