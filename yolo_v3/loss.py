import tensorflow as tf
import numpy as np
from yolo3.config import *
from decode import interprete_output


def compute_loss(scale, true, anchors, num_classes, img_size, print_loss=False):
    """
        Return yolo_loss tensor
    :param prediction: list of 3 sortie of yolo_neural_network, ko phai cua predict (N, 13, 13, 3*14)
    :param true: list(3 array) [(N,13,13,3,85), (N,26,26,3,85), (N,52,52,3,14)]
    :param anchors: array, shape=(T, 2), wh
    :param num_classes: 80
    :param img_size: list [416,416]
    :return: loss
    """
    loss = 0

    for l in range(3):

        dim = grid_shapes[l] * grid_shapes[l]

        # 预测值
        # raw_pred shape(-1,dim,3,85)
        x_y_offset, raw_pred, pred_box = interprete_output(l, scale[l], num_classes, img_size, cal_loss=True)

        # 真实值
        # true->(-1,dim,3,85)
        true[l] = tf.reshape(true[l], [-1, dim, 3, (5 + num_classes)])
        # object_mask->(-1,dim,3,1)
        true_xy, true_wh, object_mask, true_class_probs = tf.split(true[l],
                                                                   [2, 2, 1, num_classes], axis=-1)
        # Filter the box
        true_box = tf.concat([true_xy, true_wh], axis=-1)
        # iou_pred_true ->(-1,dim,3)
        iou_pred_true = cal_iou(pred_box, true_box)
        # shape(-1,dim,1)
        max_iou = tf.reduce_max(iou_pred_true, axis=-1, keep_dims=True)
        # shape(-1,dim,3)
        object_mask = tf.cast((iou_pred_true >= max_iou), tf.float32) * tf.reshape(object_mask, [-1, dim, 3])

        # 真实坐标由相对于图左上角（c_xy）变为相对于网格左上角（t_xy），与网络输出结果raw_pred（t_xy）一致
        raw_true_xy = true_xy * grid_shapes - x_y_offset
        # b_wh（0~1）->t_wh（0~1），与raw_pred中的一致
        raw_true_wh = tf.log(true_wh / anchors[anchor_mask[l]] * input_shape)
        # box_loss_scale->(-1,dim,3)
        box_loss_scale = 2 - true_wh[0] * true_wh[1]

        # loss函数
        # K.binary_crossentropy is helpful to avoid exp overflow.
        # xy_delta ->(-1,dim,3)
        xy_delta = object_mask * box_loss_scale * tf.losses.sigmoid_cross_entropy(raw_true_xy, raw_pred[..., 0:2])
        wh_delta = object_mask * box_loss_scale * 0.5 * tf.square(raw_true_wh - raw_pred[..., 2:4])
        confs_delta = object_mask * tf.losses.sigmoid_cross_entropy(object_mask, raw_pred[..., 4:5]) + \
                      (1 - object_mask) * tf.losses.sigmoid_cross_entropy(object_mask, raw_pred[..., 4:5])
        class_delta = object_mask * tf.losses.sigmoid_cross_entropy(true_class_probs, raw_pred[..., 5:])

        xy_loss = tf.reduce_mean(tf.reduce_sum(xy_delta, axis=[1, 2]))
        wh_loss = tf.reduce_mean(tf.reduce_sum(wh_delta, axis=[1, 2]))
        confidence_loss = tf.reduce_mean(tf.reduce_sum(confs_delta, axis=[1, 2]))
        class_loss = tf.reduce_mean(tf.reduce_sum(class_delta, axis=[1, 2]))

        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss],
                            message='loss: ')
    return loss


# box shape(-1,dim,3,4)
def cal_iou(box1, box2):
    box1 = tf.stack([box1[..., 0] - 0.5 * box1[..., 2],
                     box1[..., 1] - 0.5 * box1[..., 3],
                     box1[..., 0] + 0.5 * box1[..., 2],
                     box1[..., 1] + 0.5 * box1[..., 3]], axis=-1)

    box2 = tf.stack([box2[..., 0] - 0.5 * box2[..., 2],
                     box2[..., 1] - 0.5 * box2[..., 3],
                     box2[..., 0] + 0.5 * box2[..., 2],
                     box2[..., 1] + 0.5 * box2[..., 3]], axis=-1)

    lu = tf.maximum(box1[..., :2], box2[..., :2])
    rd = tf.minimum(box1[..., 2:], box2[..., 2:])

    intersection = tf.maximum(rd - lu, 0)
    over_area = intersection[..., 0] * intersection[..., 1]

    union_area = tf.maximum(box1[..., 2] * box1[..., 3] + box2[..., 2] * box2[..., 3] - over_area, 1e-10)

    return tf.clip_by_value(over_area / union_area, 0, 1)
