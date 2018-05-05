# TensorFlow CNN 表情识别模块
## 介绍传送门
[https://zijiaw.github.io/2018/05/04/Tensorflow/Tensorflow实现CNN表情识别方案/#more](https://zijiaw.github.io/2018/05/04/Tensorflow%E5%AE%9E%E7%8E%B0CNN%E8%A1%A8%E6%83%85%E8%AF%86%E5%88%AB%E6%96%B9%E6%A1%88/#more)
## 文件
### FER.py
模型训练，测试的全部内容，运行后将在同目录下生成保存好的表情识别模型，即saved_model.xxx文件。
### CNN_MODEL.py
调用表情识别模块的API，使用方法：
```py
import CNN_MODEL as cnn
sess = cnn.Initialize()
cnn.Predict(image_pixels, sess)
sess.close()
```
其中image_pixels为48*48的灰度像素值组成的numpy数组，模型文件必须在同目录下。
### face.py
可以直接运行，调用opencv打开笔记本摄像头，使用同目录下的haarcascade_frontalface_default.xml文件保存的Haar人脸检测器进行人脸定位，处理成48*48像素后调用CNN_MODEL进行人脸识别，从而在控制台显示目前的表情。