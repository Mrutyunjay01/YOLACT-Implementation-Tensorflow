import numpy as np
from sklearn.metrics import jaccard_similarity_score
import tensorflow as tf


def fast_nms(boxes, masks, scores, conf_threshold, iou_threshold: float = 0.5, top_K: int = 200,
             second_threshold: bool = False):
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_K].contiguous()
    scores = scores[:, :top_K]

    num_classes, num_dets = idx.size()

    boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = jaccard_similarity_score(boxes, boxes)
    iou = np.triu(iou, k=0)
    iou_max = iou.max(axis=-1)

    keep = (iou_max <= iou_threshold)

    if second_threshold:
        keep *= (scores * conf_threshold)
    classes = tf.keras.backend.arange(num_classes)[:None]
    classes = tf.broadcast_to(classes, keep)
    classes = classes[keep]

    boxes = boxes[keep]
    masks = masks[keep]
    scores = scores[keep]

    scores, idx = scores.sort(0, descending=True)
    idx = idx
