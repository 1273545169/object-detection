# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from yolo3.config import _ANCHORS

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1


def darknet53(inputs):
    """
    Builds Darknet-53 model.
    """
    inputs = _conv2d_fixed_padding(inputs, 32, 3)
    inputs = _conv2d_fixed_padding(inputs, 64, 3, strides=2)
    inputs = _darknet53_block(inputs, 32)
    inputs = _conv2d_fixed_padding(inputs, 128, 3, strides=2)

    for i in range(2):
        inputs = _darknet53_block(inputs, 64)

    inputs = _conv2d_fixed_padding(inputs, 256, 3, strides=2)

    for i in range(8):
        inputs = _darknet53_block(inputs, 128)

    route_1 = inputs
    inputs = _conv2d_fixed_padding(inputs, 512, 3, strides=2)

    for i in range(8):
        inputs = _darknet53_block(inputs, 256)

    route_2 = inputs
    inputs = _conv2d_fixed_padding(inputs, 1024, 3, strides=2)

    for i in range(4):
        inputs = _darknet53_block(inputs, 512)

    return route_1, route_2, inputs


def _conv2d_fixed_padding(inputs, filters, kernel_size, strides=1):
    if strides > 1:
        # # we just need to pad with one pixel, so we set kernel_size = 3
        # inputs = tf.pad(inputs, [[0, 0], [0, 2],
        #                                 [0, 2], [0, 0]])
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides, padding=('SAME' if strides == 1 else 'VALID'))
    return inputs


def _darknet53_block(inputs, filters):
    shortcut = inputs
    inputs = _conv2d_fixed_padding(inputs, filters, 1)
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)

    inputs = inputs + shortcut
    return inputs


@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, *args, mode='CONSTANT', **kwargs):
    """
    Pads the input along the spatial dimensions independently of input size.
    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      data_format: The input format ('NHWC' or 'NCHW').
      mode: The mode for tf.pad.
    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])

    return padded_inputs


def _yolo_block(inputs, filters, num_classes):
    inputs = _conv2d_fixed_padding(inputs, filters, 1)
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)
    inputs = _conv2d_fixed_padding(inputs, filters, 1)
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)
    inputs = _conv2d_fixed_padding(inputs, filters, 1)
    route = inputs
    inputs = _conv2d_fixed_padding(inputs, filters * 2, 3)
    scale = slim.conv2d(inputs, 3 * (5 + num_classes), 1, stride=1, normalizer_fn=None,
                        activation_fn=None, biases_initializer=tf.zeros_initializer())
    return route, scale


# 采用最近邻插值算法进行上采样
def _upsample(inputs, out_shape):
    inputs = tf.image.resize_nearest_neighbor(inputs, (out_shape[1], out_shape[2]))
    inputs = tf.identity(inputs, name='upsampled')
    return inputs


def yolo_v3(inputs, num_classes, is_training=False, data_format='NHWC', reuse=False):
    """
    Creates YOLO v3 model.

    :param inputs: a 4-D tensor of size [batch_size, height, width, channels].
        Dimension batch_size may be undefined. The channel order is RGB.
    :param num_classes: number of predicted classes.
    :param is_training: whether is training or not.
    :param data_format: data format NCHW or NHWC.
    :param reuse: whether or not the network and its variables should be reused.
    :return:
    """

    # set batch norm params
    batch_norm_params = {
        'decay': _BATCH_NORM_DECAY,
        'epsilon': _BATCH_NORM_EPSILON,
        'scale': True,
        'is_training': is_training,
        'fused': None,  # Use fused batch norm if possible.
    }

    # Set activation_fn and parameters for conv2d, batch_norm.
    with slim.arg_scope([slim.conv2d, slim.batch_norm, _fixed_padding], reuse=reuse):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                            biases_initializer=None, activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):
            with tf.variable_scope('darknet-53'):
                route_1, route_2, inputs = darknet53(inputs)

            with tf.variable_scope('yolo-v3'):
                # feature map1 大目标，大anchor
                route, scale1 = _yolo_block(inputs, 512, num_classes)

                # feature map2 中目标，中anchor
                inputs = _conv2d_fixed_padding(route, 256, 1)
                upsample_size = route_2.get_shape().as_list()
                inputs = _upsample(inputs, upsample_size)
                inputs = tf.concat([inputs, route_2], axis=-1)

                route, scale2 = _yolo_block(inputs, 256, num_classes)

                # feature map3 小目标，小anchor
                inputs = _conv2d_fixed_padding(route, 128, 1)
                upsample_size = route_1.get_shape().as_list()
                inputs = _upsample(inputs, upsample_size)
                inputs = tf.concat([inputs, route_1], axis=-1)

                _, scale3 = _yolo_block(inputs, 128, num_classes)

                scale = [scale1, scale2, scale3]

                return scale
