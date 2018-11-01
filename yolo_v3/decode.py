import tensorflow as tf
import numpy as np
from yolo3.config import _ANCHORS, anchor_mask
import cv2
import colorsys
import random


def interprete_output(scale_num, scale, num_classes, img_size, cal_loss=False):
    anchors = np.array(_ANCHORS).reshape(-1, 2)[anchor_mask[scale_num]]
    num_anchors = len(anchors)

    # grid_size中保存feature map的宽和高
    grid_size = scale[scale_num].shape.as_list()[1:3]
    # dim为feature map的像素点的个数
    dim = grid_size[0] * grid_size[1]
    # (5 + num_classes)= 85
    bbox_attrs = 5 + num_classes

    # stride 下采样倍数，image_size=416，grid_size=13、26、52，stride=32、16、8
    stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])
    # anchor/stride
    anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]

    # predictions：shape(-1,dim*3,85)
    raw_pred = tf.reshape(scale[scale_num], [-1, num_anchors * dim, bbox_attrs])
    box_centers, box_sizes, confidence, classes = tf.split(raw_pred, [2, 2, 1, num_classes], axis=-1)

    with tf.variable_scope("get_offset"):
        grid_x = tf.range(grid_size[0], dtype=tf.float32)
        grid_y = tf.range(grid_size[1], dtype=tf.float32)
        # a->shape(h,w)
        a, b = tf.meshgrid(grid_x, grid_y)

        # shape(dim,1)
        x_offset = tf.reshape(a, (-1, 1))
        y_offset = tf.reshape(b, (-1, 1))
        # x_y_offset：shape(dim,2)
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        # (dim,2*3)->shape(1,dim*3,2)
        x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])

    # box_centers->shape(-1,dim*3,2)
    box_centers = tf.nn.sigmoid(box_centers)
    box_centers = box_centers + x_y_offset
    # 在原始图像（416,416）中的位置
    box_centers = box_centers * stride

    # anchors->shape(dim*3,2)
    anchors = tf.tile(anchors, [dim, 1])
    # box_sizes->shape(-1,dim*3,2)
    box_sizes = tf.exp(box_sizes) * anchors
    # 在原始图像（416,416）中的大小
    box_sizes = box_sizes * stride

    # confidence->shape(-1,dim*3,1)
    confidence = tf.nn.sigmoid(confidence)
    # 类预测用sigmoid代替softmax，->shape(-1,dim*3,80)
    classes = tf.nn.sigmoid(classes)

    # shape(-1,dim*3,85)
    pred = tf.concat([box_centers, box_sizes, confidence, classes], axis=-1)

    if cal_loss:
        pred_box = tf.concat([box_centers, box_sizes], axis=-1)
        x_y_offset = tf.reshape(x_y_offset, [-1, dim, 3, 2])
        raw_pred = tf.reshape(raw_pred, [-1, dim, 3, 85])
        pred_box = tf.reshape(pred_box, [-1, dim, 3, 4])
        return x_y_offset, raw_pred, pred_box

    return pred


# 对最终预测值进行处理
# grid表示feature map的像素点的个数
def decode_output(scale, num_classes, img_size):
    predictions = []

    # Tensor("detector/yolo-v3/Conv_6/BiasAdd:0", shape=(?, 13, 13, 255), dtype = float32)
    # Tensor("detector/yolo-v3/Conv_14/BiasAdd:0", shape=(?, 26, 26, 255), dtype = float32)
    # Tensor("detector/yolo-v3/Conv_22/BiasAdd:0", shape=(?, 52, 52, 255), dtype = float32)

    for i in range(3):
        print(scale[i])

        pred = interprete_output(i, scale, num_classes, img_size)

        predictions.append(pred)

    predictions = tf.concat(predictions, axis=1)

    return predictions


# shape->(-1,dim*3,85)
def correct_boxes(detections):
    """
    Converts center x, center y, width and height values to coordinates of top left and bottom right points.

    :param detections: outputs of YOLO v3 detector of shape (?, 10647, (num_classes + 5))
    :return: converted detections of same shape as input
    """
    center_x, center_y, width, height, attrs = tf.split(detections, [1, 1, 1, 1, -1], axis=-1)

    x1 = center_x - width * 0.5
    y1 = center_y - height * 0.5
    x2 = center_x + width * 0.5
    y2 = center_y + height * 0.5

    boxes = tf.concat([x1, y1, x2, y2], axis=-1)
    detections = tf.concat([boxes, attrs], axis=-1)

    return detections


# detections->shape(1,dim*3,85)->(x1,y1,x2,y2)
def _non_max_suppression(detections, classes, score_threshold, iou_threshold):
    # boxes->(1,dim*3,4)
    boxes, confs, class_probs = tf.split(detections, [4, 1, -1], axis=-1)
    # scores->(1,dim*3,80)
    scores = confs * class_probs

    confs_mask = scores >= score_threshold

    # 存储结果
    _classes_index = []
    _scores = []
    _boxes = []
    for class_in in range(len(classes)):
        class_scores = tf.boolean_mask(scores[..., class_in], confs_mask[..., class_in])  # (dim*3,1)
        class_boxes = tf.boolean_mask(boxes, confs_mask[..., class_in])  # (dim*3,4)

        nms_index = tf.image.non_max_suppression(class_boxes, class_scores, max_output_size=30,
                                                 iou_threshold=iou_threshold)

        class_scores = tf.gather(class_scores, nms_index)
        class_boxes = tf.gather(class_boxes, nms_index)
        class_index = tf.ones_like(class_scores) * class_in

        _classes_index.append(class_index)
        _boxes.append(class_boxes)
        _scores.append(class_scores)

    # Tensor("concat:0", shape=(?,), dtype=float32)
    _classes_index = tf.cast(tf.concat(_classes_index, axis=0), tf.int32).eval(session=tf.Session())
    # Tensor("concat_1:0", shape=(?, 4), dtype=float32)
    _boxes = tf.concat(_boxes, axis=0).eval(session=tf.Session())
    _scores = tf.concat(_scores, axis=0).eval(session=tf.Session())

    return len(_classes_index), _classes_index, _boxes, _scores


def _draw_box(num_object, classes_index, boxes, scores, img, input_shape, classes,output):
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / float(len(classes)), 1., 1.)
                  for x in range(len(classes))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    # h_ratio、w_ratio
    ratio = np.array(img.shape) / input_shape
    # 线条粗细
    thick = int((img.shape[0] + img.shape[1]) / 700)
    for i in range(num_object):
        x1 = int(boxes[i][0] * ratio[1])
        y1 = int(boxes[i][1] * ratio[0])
        x2 = int(boxes[i][2] * ratio[1])
        y2 = int(boxes[i][3] * ratio[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), colors[classes_index[i].tolist()], thick)
        text = "%s : %f" % (classes[classes_index[i].tolist()], scores[i])
        cv2.putText(img, text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[classes_index[i].tolist()], 2)

        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thick)
        # text = "%s : %f" % (classes[classes_index[i].tolist()], scores[i])
        # cv2.putText(img, text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * img.shape[0], (255,0,0), 2)

    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.imwrite(output, img)
