import pylab
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data

num = int(input('请输入要展示的图片数量(0~55000)：'))
#载入数据
mnist = input_data.read_data_sets(
    "/home/vbuo/m-L-0/data/fashion", one_hot=True)

#创建一个字典存储数字所对应的标签
mark = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

for i in range(num):
    #读取一个数据
    images, labels = mnist.train.next_batch(1)
    images = images.reshape(28, 28)
    plt.imshow(images)
    #展示图
    pylab.show()
    #显示物品的属性
    for j in labels:
        m = 0
        j.shape = (10)
        for n in j:
            if n != 0:
                print('图片中的物品为:', mark[m])
            else:
                m += 1