# 2015级机器学习方向课程实训

## 2.Vehicle_License_Plate_Recognition  
描述：使用**卷积神经网络**对分割好的**车牌字符**进行识别

### 我的答案

1. [将分类好的图片及其标签序号存入到TFRecord文件中。](https://github.com/m-L-0/17b-LiShuHang-2015/blob/master/Vehicle_License_Plate_Recognition/Data/Totfrecord.ipynb)

2. [读取TFRecord文件：数据解码，reshape(恢复数据形状)。shuffle_batch。然后还有归一化处理、色彩空间变化、转换为灰色图片等操作。](https://github.com/m-L-0/17b-LiShuHang-2015/blob/master/Vehicle_License_Plate_Recognition/Data/CNN.ipynb)

3. [设计卷积神经网络结构并利用卷积神经网络对汉字和字母数字分别进行训练。](https://github.com/m-L-0/17b-LiShuHang-2015/blob/master/Vehicle_License_Plate_Recognition/Data/CNN.ipynb)

4. [利用测试集对卷积神经网络进行检测，并得到识别正确率。](https://github.com/m-L-0/17b-LiShuHang-2015/blob/master/Vehicle_License_Plate_Recognition/Data/CNN.ipynb)

5. [统计每类字符的数量与比例并利用图表展示(直方图、饼状图)](https://github.com/m-L-0/17b-LiShuHang-2015/blob/master/Vehicle_License_Plate_Recognition/Data/view.ipynb)
