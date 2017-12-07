import os
import tensorflow as tf
from PIL import Image  #注意Image,后面会用到
import matplotlib.pyplot as plt
import numpy as np


# 生成整型属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 训练集和验证集分类比例
num = 0.8
# 数据所在文件夹
cwd = '/home/vbuo/m-L-1/车牌字符识别训练数据/汉字字母数字/'
#classes = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}  #人为 设定 10 类
classes = os.listdir(cwd)
writer = tf.python_io.TFRecordWriter("word_all.tfrecords")  #生成全部文件
writer0 = tf.python_io.TFRecordWriter("word_train.tfrecords")  #生成训练集
writer1 = tf.python_io.TFRecordWriter("word_validation.tfrecords")  #生成验证集

for index, name in enumerate(classes):
    class_path = cwd + name + '/'
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name  #每一个图片的地址
        print(str(name) + ' ' + str(index))
        img = Image.open(img_path)
        img = img.resize((24, 48))
        img_raw = img.tobytes()  #将图片转化为二进制格式
        
        example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image_raw': _bytes_feature(img_raw),
                    'label': _int64_feature(index)
                }))  #example对象对label和image数据进行封装
        writer.write(example.SerializeToString())  #序列化为字符串

    #将文件存为训练集和验证集
    total_number = len(os.listdir(class_path))
    for number, image in enumerate(os.listdir(class_path)):
        """   example 已经得到"""
        if (number < num * total_number):
            writer0.write(example.SerializeToString())
        else:
            writer1.write(example.SerializeToString())

writer.close()
writer0.close()
writer1.close()
print('success')