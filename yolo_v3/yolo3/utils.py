import tensorflow as tf
import numpy as np
from PIL import ImageDraw


# 加载模型权重
def load_weights(var_list, weights_file):
    """
    Loads and converts pre-trained weights.
    :param var_list: list of network variables.
    :param weights_file: name of the binary file.
    :return: list of assign ops
    """
    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)

        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:

        var1 = var_list[i]
        # conv/weights
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        if 'Conv' in var1.name.split('/')[-2]:
            # check type of next layer
            if 'BatchNorm' in var2.name.split('/')[-2]:
                # load batch norm params
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))

                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'Conv' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))

                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops

# yolo版本通用，带输出，对照输出易理解.weigths文件的结构
def load_weights1(var_list, weights_file):
    """
    Loads and converts pre-trained weights.
    :param var_list: list of network variables.
    :param weights_file: name of the binary file.
    :return: list of assign ops
    """

    with open(weights_file, "rb") as fp:
        # This is verry import for count,it include the version of yolo
        # yolo1、yolo2中count=4，yolo3中count=5
        _ = np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        # detector/darknet-53/Conv/BatchNorm/
        var1 = var_list[i]
        var2 = var_list[i + 1]
        # do something only if we process conv layer
        if 'Conv' in var1.name.split('/')[-2]:
            # check type of next layer,BatchNorm param first of weight
            if 'BatchNorm' in var2.name.split('/')[-2]:
                # load batch norm params, It's equal to l.biases,l.scales,l.rolling_mean,l.rolling_variance
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    print(var, ptr)
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))

                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'Conv' in var2.name.split('/')[-2]:
                # load biases,not use the batch norm,So just only load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                print(bias, ptr)
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
                # we loaded 1 variable
                i += 1

            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            print(var1, ptr)
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1
        elif 'Local' in var1.name.split('/')[-2]:
            # load biases
            bias = var2
            bias_shape = bias.shape.as_list()
            bias_params = np.prod(bias_shape)
            bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
            ptr += bias_params
            print(bias, ptr)
            assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
            i += 1

            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            print(var1, ptr)
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1
        elif 'Fc' in var1.name.split('/')[-2]:
            # load biases
            bias = var2
            bias_shape = bias.shape.as_list()
            bias_params = np.prod(bias_shape)
            bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
            ptr += bias_params
            print(bias, ptr)
            assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))
            i += 1

            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape(shape[1], shape[0])
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (1, 0))
            ptr += num_params
            print(var1, ptr)
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1

    return assign_ops


def save_weight_to_pbfile(sess, weight_file):

    # 导出当前图的GraphDef部分即可完成从输入层到输出层的计算过程
    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = tf.graph_util.\
        convert_variables_to_constants(sess, graph_def,
                                       output_node_names=["output_boxes","inputs"])
    with tf.gfile.GFile(weight_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())


def restore_weight_from_pbfile(weight_file):

    with tf.gfile.GFile(weight_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")

