yolo是one-stage的目标检测算法

参考资料：

1、[paper](https://pjreddie.com/media/files/papers/yolo.pdf)

2、[论文翻译](http://noahsnail.com/2017/08/02/2017-8-2-YOLO%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E8%8B%B1%E6%96%87%E5%AF%B9%E7%85%A7/)

3、[yolo详解](https://zhuanlan.zhihu.com/p/32525231)

4、[视频讲解](https://www.bilibili.com/video/av23354360/?p=1)
[ppt](https://drive.google.com/file/d/164mVbMBhoMzY5pkaEOdK3IIcIwTOj2B-/view)




## YOLO_tensorflow

包括训练部分和测试部分

### Installation

1. Clone yolo_tensorflow repository

2. Download [Pascal VOC dataset](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)
, and put in :` ./data/pascal_voc/ `

3. Download [YOLO_small](https://drive.google.com/file/d/0B5aC8pI-akZUNVFZMmhmcVRpbTA/view?usp=sharing)
weight file and put it in `./data/weight/`

4. Modify configuration in `yolo/config.py`

5. Training
	` train.py`


6. Test
   `predict.py`

### Requirements
1. Tensorflow

2. OpenCV


