### 数据分布：![](https://github.com/m-L-0/17b-LiShuHang-2015/blob/master/CaptchaRecognition/static/QQ%E6%88%AA%E5%9B%BE20180109143628.png)![](https://github.com/m-L-0/17b-LiShuHang-2015/blob/master/CaptchaRecognition/static/QQ%E6%88%AA%E5%9B%BE20180109143652.png)

### 模型的设计：
1. 网络结构采用卷积神经网络(cnn)
2. 采用端到端的网络结构 

#### 网络结构：
1. 卷积层3*3 
2. 池化层 
3. 卷积层3*3 
4. 池化层 
5. 卷积层3*3 
6. 池化层 
7. 全连接层 
8. 四个输出层 

#### 特点：
1. 卷积层比池化层的数量多
2. 卷积的卷积核的参数的高和宽不一样(卷积窗口不是正方形)
3. 使用了多个variable_scope包裹模型

### 正确率：
训练时的正确率在：97%~98%之间 
验证集的正确率到90%以上 

### web界面：
使用flask提供了web端的可视化界面。
