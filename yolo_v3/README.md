### yolo_v3的tensorflow实现
yolo系列解析：https://blog.csdn.net/baidu_27643275/article/details/82964784

[tensorflow-yolo-v3](https://github.com/mystic123/tensorflow-yolo-v3) 和 [maiminh1996/YOLOv3-tensorflow](https://github.com/maiminh1996/YOLOv3-tensorflow) 

## 完整的工作有:
- [x] YOLO_v3 网络
- [x] covert_weight（将yolo官方提供的.weight文件转换为.ckpt形式来保存权重）
- [x] 预测过程
- [x] 训练过程所用的的loss函数
- [ ] 没有完整的训练过程

## 运行:
1. Download and convert model weights:    
    1. Download binary file with weights from https://pjreddie.com/darknet/yolo/
    2. Run `python ./covert_weight.py`

2、Run `python ./predict.py`

