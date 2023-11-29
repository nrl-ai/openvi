#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import math

import cv2
import numpy as np
import onnxruntime
from scipy.special import softmax


mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3) / 255
std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3) / 255

def integral(reg_max=7):
    project = np.linspace(0, reg_max, reg_max + 1)
    def func(x):
        shape = x.shape
        x = softmax(x.reshape(*shape[:-1], 4, reg_max + 1), axis=-1)
        x = np.dot(x, project).reshape(*shape[:-1], 4)
        return x

    return func

class Nanodet(object):
    def __init__(
        self,
        model_path='model/nanodet.onnx',
        class_names=['person']*80,
        reg_max=7,
        class_score_th=0.35,
        nms_th=0.5,
        nms_score_th=0.1,
        providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    ):
        self.class_names = class_names
        self.reg_max = reg_max
        self.class_score_th = class_score_th
        self.nms_th = nms_th
        self.nms_score_th = nms_score_th

        self.onnx_session = onnxruntime.InferenceSession(
            model_path,
            providers=providers,
        )

        self.input_detail = self.onnx_session.get_inputs()[0]
        self.input_name = self.input_detail.name
        self.output_name = self.onnx_session.get_outputs()[0].name
        self.input_shape = self.input_detail.shape[2:]
        # print(self.input_detail)
        # print(self.input_shape)
    
    def __call__(self, image):
        temp_image = copy.deepcopy(image)
        image_height, image_width = image.shape[0], image.shape[1]
        image, ratio = self._preprocess(temp_image, self.input_shape)
        ort_inputs = {self.input_name: image[None, :, :, :]}
        output = self.onnx_session.run(None, ort_inputs)
        # results = self._postprocess(preds=output[0],
        #                             input_size=self.input_shape,
        #                             ratio=ratio,
        #                             num_classes=len(self.class_names),
        #                             nms_th=self.nms_th,
        #                             nms_score_th=self.nms_score_th,
        #                             reg_max=self.reg_max,
        #                             max_width=image_width,
        #                             max_height=image_height,
        #                             )
        # return results
        bboxes, scores, class_ids = self._postprocess(preds=output[0],
                            input_size=self.input_shape,
                            ratio=ratio,
                            num_classes=len(self.class_names),
                            nms_th=self.nms_th,
                            nms_score_th=self.nms_score_th,
                            reg_max=self.reg_max,
                            max_width=image_width,
                            max_height=image_height,
                            )
        return bboxes, scores, class_ids
    
    def _preprocess(self, image, input_size, swap=(2, 0, 1)):
        # Note that keep_ratio default config for train and val are set False
        # therefore, this version only supports keep_ratio=False
        # we will update keep_ratio effect in the next version
        src_height, src_width = image.shape[:2]
        h_ratio = src_height / input_size[0]
        w_ratio = src_width / input_size[1]

        resized_img = cv2.resize(image, (input_size[1], input_size[0]))
        normalized_img = resized_img.astype(np.float32) / 255
        normalized_img = (normalized_img - mean) / std
        normalized_img = normalized_img.transpose(swap)
        normalized_img = np.ascontiguousarray(normalized_img, dtype=np.float32)


        return normalized_img, [h_ratio, w_ratio]
    
    def _postprocess(
            self,
            preds,
            input_size,
            ratio,
            num_classes,
            nms_th,
            nms_score_th,
            reg_max,
            max_width,
            max_height,
            strides = [8, 16, 32, 64],
        ):
        """Prediction results post processing. Decode bboxes and rescale
        to original image size.
        Args:
        preds (Tensor): Prediction output.
        meta (dict): Meta info.
        """

        cls_scores, bbox_preds = np.split(preds, [num_classes], axis=-1)
        result_list = self._get_bboxes(cls_scores, bbox_preds, input_size, nms_th, nms_score_th, reg_max, strides)
        det_results = {}
        warp_matrix = np.eye(3)
        img_height = input_size[0]
        img_width = input_size[1]
        det_result = {}
        det_bboxes, det_labels = result_list[0][:, :5], result_list[0][:, 5]
        det_bboxes[:, :4] = self._warp_boxes(det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height)
        for i in range(num_classes):
                inds = det_labels == i
                det_result[i] = np.concatenate(
                    [
                        det_bboxes[inds, :4].astype(np.float32),
                        det_bboxes[inds, 4:5].astype(np.float32),
                    ],
                    axis=1,
                ).tolist()
        bboxes, scores, class_ids = [], [], []
        w_ratio = ratio[1]
        h_ratio = ratio[0]
        if len(det_result) != 0:
            for label in det_result:
                for bbox in det_result[label]:
                    score = bbox[-1]
                    x0, y0, x1, y1 = [int(i) for i in bbox[:4]]
                    x0 = int(x0 * w_ratio)
                    y0 = int(y0 * h_ratio)
                    x1 = int(x1 * w_ratio)
                    y1 = int(y1 * h_ratio)

                    bboxes.append([x0, y0, x1, y1])
                    scores.append(score)
                    class_ids.append(label)
        # det_results[0] = det_result
        # return det_results
        return np.asarray(bboxes), np.asarray(scores), np.asarray(class_ids)

    def _get_bboxes(self, cls_preds, reg_preds, input_shape, nms_th, nms_score_th, reg_max, strides):
        distribution_project = integral(reg_max)
        b = cls_preds.shape[0]
        input_height, input_width = input_shape

        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in strides
        ]
        # get grid cells of one image
        mlvl_center_priors = [self._get_single_level_center_priors(
                b,
                featmap_sizes[i],
                stride
            )
            for i, stride in enumerate(strides)
        ]

        center_priors = np.concatenate(mlvl_center_priors, axis=1)
        dis_preds = distribution_project(reg_preds) * center_priors[..., 2, None]
        bboxes = self._distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
        scores = cls_preds
        result_list = []
        for i in range(b):
            score, bbox = scores[i], bboxes[i]
            padding = np.zeros((score.shape[0], 1))
            score = np.concatenate([score, padding], axis=1)
            results = self._multiclass_nms_class(bbox, score, nms_thr=nms_th, score_thr=nms_score_th)
            result_list.append(results)
        return result_list
    
    def _get_single_level_center_priors(self, batch_size, featmap_size, stride):
        h, w = featmap_size
        x_range = (np.arange(w)) * stride
        y_range = (np.arange(h)) * stride
        x, y = np.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        strides = np.full((x.shape[0],), stride)
        proiors = np.stack([x, y, strides, strides], axis=-1)
        return proiors[np.newaxis, ...].repeat(batch_size, axis=0)

    def _distance2bbox(self, points, distance, max_shape=None):
        x1 = points[..., 0] - distance[..., 0]
        y1 = points[..., 1] - distance[..., 1]
        x2 = points[..., 0] + distance[..., 2]
        y2 = points[..., 1] + distance[..., 3]
        if max_shape is not None:
            x1 = np.minimum(np.maximum(x1, 0.0), max_shape[1])
            y1 = np.minimum(np.maximum(y1, 0.0), max_shape[0])
            x2 = np.minimum(np.maximum(x2, 0.0), max_shape[1])
            y2 = np.minimum(np.maximum(y2, 0.0), max_shape[0])

        return np.stack([x1, y1, x2, y2], axis=-1)

    def _multiclass_nms_class(self, boxes, scores, nms_thr, score_thr):
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self._nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
            )
        return dets
    
    def _nms(self, boxes, scores, nms_thr):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]

        return keep
    def _warp_boxes(self, boxes, M, width, height):
        n = len(boxes)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                n * 4, 2
            )  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            return xy.astype(np.float32)
        else:
            return boxes
    
    def draw(
        self,
        image,
        score_th,
        bboxes,
        scores,
        class_ids,
        coco_classes,
        thickness=3,
    ):
        debug_image = copy.deepcopy(image)

        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(
                bbox[3])

            if score_th > score:
                continue

            color = self._get_color(class_id)

            debug_image = cv2.rectangle(
                debug_image,
                (x1, y1),
                (x2, y2),
                color,
                thickness=thickness,
            )

            score = '%.2f' % score
            text = '%s:%s' % (str(coco_classes[int(class_id)]), score)
            debug_image = cv2.putText(
                debug_image,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                thickness=thickness,
            )

        return debug_image

    def _get_color(self, index):
        temp_index = abs(int(index + 5)) * 3
        color = (
            (29 * temp_index) % 255,
            (17 * temp_index) % 255,
            (37 * temp_index) % 255,
        )
        return color

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    # Load COCO Classes List
    with open('coco_classes.txt', 'rt') as f:
        coco_classes = f.read().rstrip('\n').split('\n')
    print(f"Len coco {len(coco_classes)}")
    # Load model
    model_path = 'model/nanodet.onnx'
    model = Nanodet(model_path=model_path, class_names=coco_classes, reg_max=7, nms_th=0.5, nms_score_th=0.1)

    while True:
        # Capture read
        ret, frame = cap.read()
        if not ret:
            break

        # Inference execution
        bboxes, scores, class_ids = model(frame)

        # Draw
        frame = model.draw(
            frame,
            0.3,
            bboxes,
            scores,
            class_ids,
            coco_classes,
        )

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        cv2.imshow('Nanodet', frame)
    cap.release()
    cv2.destroyAllWindows()

    ## Test image
    # frame = cv2.imread("bus.jpg")
    # bboxes, scores, class_ids = model(frame)
    # frame = model.draw(
    #         frame,
    #         0.3,
    #         bboxes,
    #         scores,
    #         class_ids,
    #         coco_classes,
    #     )
    # cv2.imwrite(f"output.jpg", frame)
    # cv2.imshow('Nanodet', frame)

