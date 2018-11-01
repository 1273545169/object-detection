# -*- coding: utf-8 -*-
from yolo3.utils import *
from yolo3.config import *
from yolo3.yolo_v3 import yolo_v3
from decode import *
from decode import _non_max_suppression, _draw_box

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_img', './test/car.jpg', '')
tf.app.flags.DEFINE_string('output_img', './test/Output.jpg', 'Output image')
tf.app.flags.DEFINE_string('class_names', './data/coco_names.txt', 'File with class names')
tf.app.flags.DEFINE_string('weights_file', './data/weights/yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string('ckpt_file', "data/checkpoint/yolov3.ckpt", 'Binary file with detector weights')

tf.app.flags.DEFINE_integer('size', 416, 'Image size')

tf.app.flags.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')


def main(argv=None):
    classes = load_coco_names(FLAGS.class_names)

    with tf.variable_scope("prepross_image"):
        img = cv2.imread(FLAGS.input_img)
        inputs = cv2.resize(img, (FLAGS.size, FLAGS.size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = np.reshape(inputs, [1, FLAGS.size, FLAGS.size, 3])
        # normalize values to range [0..1]
        inputs = inputs / 255

    with tf.variable_scope('detector'):
        # placeholder for detector inputs
        inputs_placeholder = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3])
        scale = yolo_v3(inputs_placeholder, len(classes), data_format='NHWC')
        img_size = inputs_placeholder.shape.as_list()[1:3]
        detections = decode_output(scale, len(classes), img_size)
        boxes = correct_boxes(detections)

    with tf.Session() as sess:
        # # 从.weight文件中加载权重
        # load_ops = load_weights(tf.global_variables(scope='detector'), FLAGS.weights_file)
        # sess.run(load_ops)

        # 从ckpt文件中加载权重
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.ckpt_file)

        detected_boxes = sess.run(boxes, feed_dict={inputs_placeholder: inputs})

        num_object, classes_in, boxes, scores = _non_max_suppression(detected_boxes, classes,
                                                                     score_threshold=FLAGS.conf_threshold,
                                                                     iou_threshold=FLAGS.iou_threshold)

        print("{}个目标：{}".format(num_object, np.array(classes)[classes_in]))

        _draw_box(num_object, classes_in, boxes, scores, img, FLAGS.size, classes,FLAGS.output_img)


if __name__ == '__main__':
    tf.app.run()
