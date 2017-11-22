import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


# 把传入的value转化为整数型的属性，int64_list对应着 tf.train.Example 的定义
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 把传入的value转化为字符串型的属性，bytes_list对应着 tf.train.Example 的定义
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


#读取fashion-mnist训练集数据
fashionmnist = input_data.read_data_sets(
    "../m-L-0/data/fashion",
    dtype=tf.uint8,
    one_hot=True)
#训练数据的图像,作为一个属性来存储
images = fashionmnist.train.images
#训练数据所对应的正确答案,作为一个属性来存储
labels = fashionmnist.train.labels
#训练数据的图像分辨率,可以作为一个属性来存储
pixels = images.shape[0]
#训练数据的个数
num_examples = fashionmnist.train.num_examples
#写入TFRecord文件的地址
filename = "../m-L-0/CC/train.tfrecords"
#创建一个writer来写TFRecord文件
writer = tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
    #把图像矩阵转化为字符串
    image_raw = images[index].tostring()
    #将一个样例转化为Example Protocol Buffer，并将所有的信息写入这个数据结构
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'pixels': _int64_feature(pixels),
            'label': _int64_feature(np.argmax(labels[index])),
            'image_raw': _bytes_feature(image_raw)
        }))
    #将 Example 写入TFRecord文件
    writer.write(example.SerializeToString())
writer.close()

#读取fashion-mnist测试集数据
fashionmnist = input_data.read_data_sets(
    "../m-L-0/data/fashion",
    dtype=tf.uint8,
    one_hot=True)
#测试数据的图像,作为一个属性来存储
images = fashionmnist.test.images
#测试数据所对应的正确答案,作为一个属性来存储
labels = fashionmnist.test.labels
#测试数据的图像分辨率,作为一个属性来存储
pixels = images.shape[0]
#测试数据的个数
num_examples = fashionmnist.test.num_examples
#写入TFRecord文件的地址
filename = "../m-L-0/CC/test.tfrecords"
#创建一个writer来写TFRecord文件  无标题文档
writer = tf.python_io.TFRecordWriter(filename)
for index in range(num_examples):
    #把图像矩阵转化为字符串
    image_raw = images[index].tostring()
    #将一个样例转化为Example Protocol Buffer，并将所有的信息写入这个数据结构
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'pixels': _int64_feature(pixels),
            'label': _int64_feature(np.argmax(labels[index])),
            'image_raw': _bytes_feature(image_raw)
        }))
    #将 Example 写入TFRecord文件
    writer.write(example.SerializeToString())
writer.close()
print('数据转换成功')
