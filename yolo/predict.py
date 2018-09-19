# 模型预测

import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
import time


class Detector(object):
    def __init__(self, net, weight_file):
        self.net = net
        self.weights_file = weight_file

        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.threshold = cfg.THRESHOLD
        self.iou_threshold = cfg.IOU_THRESHOLD
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + \
                         self.cell_size * self.cell_size * self.boxes_per_cell

        self.sess = tf.Session()
        # 加载权重
        self._load_weights(self.weights_file)

    def _load_weights(self, weights_file):
        """Load weights from file"""
        print('Restoring weights from: ' + weights_file)
        saver = tf.train.Saver()
        saver.restore(self.sess, weights_file)

    def detect(self, image_file):
        img = cv2.imread(image_file)
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})

        result = self.interpret_output(net_output[0])

        print(result)

        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.image_size)
            result[i][2] *= (1.0 * img_h / self.image_size)
            result[i][3] *= (1.0 * img_w / self.image_size)
            result[i][4] *= (1.0 * img_h / self.image_size)

        self.draw_result(img, result)

    def interpret_output(self, output):

        class_probs = np.reshape(
            output[0:self.boundary1],
            (self.cell_size, self.cell_size, self.num_class))
        confs = np.reshape(
            output[self.boundary1:self.boundary2],
            (self.cell_size, self.cell_size, self.boxes_per_cell))
        boxes = np.reshape(
            output[self.boundary2:],
            (self.cell_size, self.cell_size, self.boxes_per_cell, 4))

        x_offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
                                           [self.boxes_per_cell, self.cell_size, self.cell_size]), [1, 2, 0])
        y_offset = np.transpose(x_offset, [1, 0, 2])

        # convert the x, y to the coordinates relative to the top left point of the image
        # the predictions of w, h are the square root
        # multiply the width and height of image
        boxes = tf.stack([(boxes[:, :, :, 0] + tf.constant(x_offset, dtype=tf.float32)) / self.cell_size * self.image_size,
                          (boxes[:, :, :, 1] + tf.constant(y_offset, dtype=tf.float32)) / self.cell_size * self.image_size,
                          tf.square(boxes[:, :, :, 2]) * self.image_size,
                          tf.square(boxes[:, :, :, 3]) * self.image_size], axis=3)

        # 对bounding box的筛选分别三步进行
        # 第一步：求得每个bounding box所对应的最大的confidence，结果有7*7*2个
        # 第二步：根据confidence threshold来对bounding box筛选
        # 第三步：NMS

        # shape(7,7,2,20)
        class_confs = tf.expand_dims(confs, -1) * tf.expand_dims(class_probs, 2)

        # 4维变2维  shape(7*7*2,20)
        class_confs = tf.reshape(class_confs, [-1, self.num_class])
        # shape(7*7*2,4)
        boxes = tf.reshape(boxes, [-1, 4])

        # 第一步：find each box class, only select the max confidence
        # 求得每个bounding box所对应的最大的class confidence，有7*7*2个bounding box，所以个结果有98个
        class_index = tf.argmax(class_confs, axis=1)
        class_confs = tf.reduce_max(class_confs, axis=1)

        # 第二步：filter the boxes by the class confidence threshold
        filter_mask = class_confs >= self.threshold
        class_index = tf.boolean_mask(class_index, filter_mask)
        class_confs = tf.boolean_mask(class_confs, filter_mask)
        boxes = tf.boolean_mask(boxes, filter_mask)

        # 第三步: non max suppression (do not distinguish different classes)
        # 一个目标可能有多个预测框，通过NMS可以去除多余的预测框，确保一个目标只有一个预测框
        # box (x, y, w, h) -> nms_boxes (x1, y1, x2, y2)
        nms_boxes = tf.stack([boxes[:, 0] - 0.5 * boxes[:, 2], boxes[:, 1] - 0.5 * boxes[:, 3],
                              boxes[:, 0] + 0.5 * boxes[:, 2], boxes[:, 1] + 0.5 * boxes[:, 3]], axis=1)
        # NMS:
        # 先将class_confs按照降序排列，然后计算第一个confs所对应的box与其余box的iou，
        # 若大于iou_threshold，则将其余box的值设为0。
        nms_index = tf.image.non_max_suppression(nms_boxes, class_confs,
                                                 max_output_size=10,
                                                 iou_threshold=self.iou_threshold)

        class_index = tf.gather(class_index, nms_index)
        class_confs = tf.gather(class_confs, nms_index)
        boxes = tf.gather(boxes, nms_index)
        # tensor -> numpy，因为tensor中没有len()方法
        class_index = class_index.eval(session=self.sess)
        class_confs = class_confs.eval(session=self.sess)
        boxes = boxes.eval(session=self.sess)

        result = []
        for i in range(len(class_index)):
            result.append([self.classes[class_index[i]],
                           boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                           class_confs[i]])


        return result

    def draw_result(self, img, result):
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1)
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
            cv2.putText(
                img, result[i][0] + ' : %.2f' % result[i][5],
                (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, lineType)

        cv2.imshow('Image', img)
        cv2.waitKey(1)
        cv2.imwrite('test/predict_result.jpg',img)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--weight_dir', default='weights', type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    yolo = YOLONet(False)
    weight_file = os.path.join(args.data_dir, args.weight_dir, args.weights)
    detector = Detector(yolo, weight_file)

    # detect from image file
    image_file = 'test/cat.jpg'
    detector.detect(image_file)



if __name__ == '__main__':
    main()
