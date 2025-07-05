import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# 리엑트 네이티브 컨버팅 용도로 구현한 포즈 추정 클래스
# 미디어파이프 포즈 디텍션 c++ 코드 변환
class PoseDetectionTFLITE(object):

    # 모바일 필수 컨버팅
    def __init__(self,
                 model_path="fdlite/data/pose_detection.tflite"):
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.opts = {
            'num_layers': 5,
            'input_size_height': 224,
            'input_size_width': 224,
            'anchor_offset_x': 0.5,
            'anchor_offset_y': 0.5,
            'min_scale': 0.1484375,
            'max_scale': 0.75,
            'strides': [8, 16, 32, 32, 32],
            'fixed_anchor_size': True
            # If using fixed_anchor_size=False, ensure to add 'aspect_ratios': [1.0]
        }
        self.anchor = self._ssd_generate_anchors()

    # 모바일 필수 컨버팅
    def inference(self, input_tensor):
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        raw_boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        raw_scores = self.interpreter.get_tensor(self.output_details[1]['index'])
        return raw_boxes, raw_scores

    def _ssd_generate_anchors(self) -> np.ndarray:
        """Generate anchors consistent with MediaPipe implementation, with support for fixed_anchor_size."""
        num_layers = self.opts['num_layers']
        strides = self.opts['strides']
        assert len(strides) == num_layers
        input_height = self.opts['input_size_height']
        input_width = self.opts['input_size_width']
        anchor_offset_x = self.opts['anchor_offset_x']
        anchor_offset_y = self.opts['anchor_offset_y']
        min_scale = self.opts['min_scale']
        max_scale = self.opts['max_scale']
        scales = [self._calculate_scale(min_scale, max_scale, i, num_layers) for i in range(num_layers)]

        # Extract fixed_anchor_size option
        fixed_anchor_size = self.opts.get('fixed_anchor_size', True)

        anchors = []

        for layer_id in range(num_layers):
            stride = strides[layer_id]
            feature_map_height = int(np.ceil(input_height / stride))
            feature_map_width = int(np.ceil(input_width / stride))

            for y in range(feature_map_height):
                y_center = (y + anchor_offset_y) / feature_map_height
                for x in range(feature_map_width):
                    x_center = (x + anchor_offset_x) / feature_map_width
                    if fixed_anchor_size:
                        if layer_id == 0 and self.opts.get("reduce_boxes_in_lowest_layer", False):
                            aspect_ratios = [1.0, 2.0, 0.5]
                            scales_for_layer = [0.1, scales[layer_id], scales[layer_id]]
                        else:
                            aspect_ratios = self.opts.get('aspect_ratios', [1.0])
                            scales_for_layer = [scales[layer_id]] * len(aspect_ratios)
                            interpolated_ratio = self.opts.get('interpolated_scale_aspect_ratio', 1.0)
                            if interpolated_ratio > 0.0:
                                scale_next = scales[layer_id + 1] if layer_id < num_layers - 1 else 1.0
                                interpolated_scale = (scales[layer_id] * scale_next) ** 0.5
                                aspect_ratios.append(interpolated_ratio)
                                scales_for_layer.append(interpolated_scale)

                        for _ in range(len(aspect_ratios)):
                            anchors.append((x_center, y_center, 1.0, 1.0))
                    else:
                        import math
                        aspect_ratios = self.opts.get('aspect_ratios', [1.0])
                        scale = scales[layer_id]
                        scale_next = scales[layer_id + 1] if layer_id < num_layers - 1 else 1.0
                        interpolated_scale = math.sqrt(scale * scale_next)
                        for ratio in aspect_ratios:
                            ratio_sqrt = math.sqrt(ratio)
                            anchor_w = scale * ratio_sqrt
                            anchor_h = scale / ratio_sqrt
                            anchors.append((x_center, y_center, anchor_w, anchor_h))
                        # Add interpolated anchor with aspect ratio 1.0
                        anchors.append((x_center, y_center, interpolated_scale, interpolated_scale))
        return np.array(anchors, dtype=np.float32)

    # 모바일 필수 컨버팅
    def preprocess(self, image):
        self.image_draw = np.array(image).copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        target_w = self.opts['input_size_width']
        target_h = self.opts['input_size_height']

        self.scale = min(target_w / w, target_h / h)
        resized_w, resized_h = int(w * self.scale), int(h * self.scale)
        resized = cv2.resize(image, (resized_w, resized_h))
        input_tensor = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        input_tensor[:resized_h, :resized_w, :] = resized

        input_tensor = input_tensor.astype(np.float32)
        input_tensor = (input_tensor / 127.5) - 1.0
        return np.expand_dims(input_tensor, axis=0)

    @staticmethod
    def _calculate_scale(min_scale, max_scale, stride_index, num_strides):
        if num_strides == 1:
            return (min_scale + max_scale) * 0.5
        else:
            return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1)

    @staticmethod
    def _decode_boxes(raw_boxes, anchors, apply_exponential_on_box_size=False) -> np.ndarray:
        """Decode raw box predictions to bounding boxes using MediaPipe's scale."""
        x_scale = y_scale = w_scale = h_scale = 224.0
        num_boxes = raw_boxes.shape[0]

        x_center = raw_boxes[:, 0] / x_scale * anchors[:, 2] + anchors[:, 0]
        y_center = raw_boxes[:, 1] / y_scale * anchors[:, 3] + anchors[:, 1]
        if apply_exponential_on_box_size:
            w = np.exp(raw_boxes[:, 2] / w_scale) * anchors[:, 2]
            h = np.exp(raw_boxes[:, 3] / h_scale) * anchors[:, 3]
        else:
            w = raw_boxes[:, 2] / w_scale * anchors[:, 2]
            h = raw_boxes[:, 3] / h_scale * anchors[:, 3]

        xmin = x_center - w / 2
        ymin = y_center - h / 2
        xmax = x_center + w / 2
        ymax = y_center + h / 2

        # Return in [N, 2, 2] format to match expected (xmin/ymin, xmax/ymax) use
        return np.stack([np.stack([xmin, ymin], axis=1),
                         np.stack([xmax, ymax], axis=1)], axis=1)

    @staticmethod
    def _decode_keypoints(raw_boxes, anchors, x_scale=224.0, y_scale=224.0):
        """Decode 4 keypoints using MediaPipe's linear decoding."""
        keypoints = np.zeros((raw_boxes.shape[0], 4, 2), dtype=np.float32)
        for i in range(4):
            offset = 4 + i * 2
            keypoint_x = raw_boxes[:, offset]
            keypoint_y = raw_boxes[:, offset + 1]
            x = keypoint_x / x_scale * anchors[:, 2] + anchors[:, 0]
            y = keypoint_y / y_scale * anchors[:, 3] + anchors[:, 1]
            keypoints[:, i] = np.stack([x, y], axis=-1)
        return keypoints

    @staticmethod
    def _compute_rotated_bbox_from_keypoints(kpt0, kpt1, image_w, image_h, scale=1.25):
        x0, y0 = kpt0
        x1, y1 = kpt1
        cx = x0 * image_w
        cy = y0 * image_h
        dx = x1 * image_w - cx
        dy = -(y1 * image_h - cy)  # Y축 반전
        angle = np.degrees(np.arctan2(dy, dx)) - 90
        distance = np.sqrt(dx ** 2 + dy ** 2)
        box_size = distance * 2.0 * scale
        return cx, cy, box_size, box_size, angle

    # 모바일 필수 컨버팅(점수 분기용. 대신 argmax 한 값 하나에만 적용해서 계산량 줄이자)
    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x))

    # 인퍼런스하는것만 컨버팅하면 될듯
    def run(self, image, draw=True, threshold=0.5):
        input_tensor = self.preprocess(image)
        raw_boxes, raw_scores = self.inference(input_tensor)

        # best 1 추출하기
        raw_scores = self._sigmoid(raw_scores)
        best_idx = np.argmax(raw_scores[0, :, 0])

        # best 1 기준 raw_boxes 가져오기
        selected_box = raw_boxes[0, best_idx:best_idx+1]
        selected_anchor = self.anchor[best_idx:best_idx+1]
        self.best_score = raw_scores[0, best_idx, 0]

        # 해당 박스에 _decode_boxes, _decode_keypoints 적용하기
        decoded_bbox = self._decode_boxes(selected_box, selected_anchor)
        decoded_kpts = self._decode_keypoints(selected_box, selected_anchor)

        self.best_bbox = decoded_bbox[0]
        self.best_keypoints = decoded_kpts[0]

        if draw & (self.best_score >= threshold):
            h, w, _ = self.image_draw.shape
            input_h, input_w = input_tensor.shape[1:3]
            scale = self.scale

            (xmin, ymin), (xmax, ymax) = self.best_bbox

            xmin = xmin * input_w
            xmax = xmax * input_w
            ymin = ymin * input_h
            ymax = ymax * input_h
            xmin = int(xmin / scale)
            xmax = int(xmax / scale)
            ymin = int(ymin / scale)
            ymax = int(ymax / scale)
            xmin = int(np.clip(xmin, 0, w))
            xmax = int(np.clip(xmax, 0, w))
            ymin = int(np.clip(ymin, 0, h))
            ymax = int(np.clip(ymax, 0, h))
            cv2.rectangle(self.image_draw, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            kpt0 = self.best_keypoints[0]
            kpt2 = self.best_keypoints[2]
            cx0 = int(np.clip((kpt0[0] * input_w) / scale, 0, w))
            cy0 = int(np.clip((kpt0[1] * input_h) / scale, 0, h))
            cx2 = int(np.clip((kpt2[0] * input_w) / scale, 0, w))
            cy2 = int(np.clip((kpt2[1] * input_h) / scale, 0, h))
            cv2.circle(self.image_draw, (cx0, cy0), 4, (255, 0, 0), -1)
            cv2.circle(self.image_draw, (cx2, cy2), 4, (255, 0, 0), -1)

            kpt1 = self.best_keypoints[1]
            cx, cy, box_w, box_h, angle = self._compute_rotated_bbox_from_keypoints(kpt0, kpt1, w, h)
            rect = ((cx, cy), (box_w, box_h), angle)
            box = cv2.boxPoints(rect).astype(np.int32)
            cv2.polylines(self.image_draw, [box], isClosed=True, color=(0, 165, 255), thickness=2)

        return self.best_bbox, self.best_score, self.image_draw