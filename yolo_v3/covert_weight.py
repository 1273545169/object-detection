# -*- coding: utf-8 -*-
from yolo3.utils import *
from yolo3.yolo_v3 import yolo_v3
from yolo3.config import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('weights_file', './data/weights/yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string('ckpt_file', "data/checkpoint/yolov3.ckpt", 'Binary file with detector weights')
tf.app.flags.DEFINE_string('pb_file', "./data/pb/yolo3.pb", 'Binary file with detector weights')

tf.app.flags.DEFINE_integer('size', 416, 'Image size')


def main(argv=None):

    with tf.variable_scope('detector'):

        inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3])
        scale = yolo_v3(inputs, num_class, data_format='NHWC')

    with tf.Session() as sess:

        load_ops = load_weights1(tf.global_variables(scope='detector'), FLAGS.weights_file)
        sess.run(load_ops)

        # 将权重保存为ckpt文件
        saver=tf.train.Saver()
        saver.save(sess,FLAGS.ckpt_file)

        # 将权重保存为.pb文件
        # save_weight_to_pbfile(sess,FLAGS.pb_file)

if __name__ == '__main__':
    tf.app.run()
