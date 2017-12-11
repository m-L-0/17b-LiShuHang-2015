import os
import tensorflow as tf
from PIL import Image
Label = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'A': 10,
    'B': 11,
    'C': 12,
    'D': 13,
    'E': 14,
    'F': 15,
    'G': 16,
    'H': 17,
    'J': 18,
    'K': 19,
    'L': 20,
    'M': 21,
    'N': 22,
    'P': 23,
    'Q': 24,
    'R': 25,
    'S': 26,
    'T': 27,
    'U': 28,
    'V': 29,
    'W': 30,
    'X': 31,
    'Y': 32,
    'Z': 33
}

# 数据所在文件夹
cwd = '/home/vbuo/m-L-1/车牌字符识别训练数据/'
classdir = {'数字', '字母'}
# 获取所有文件夹中包含文件数最少的文件夹的文件数量，每个文件夹取此文件夹中75%的图片数量，即152张
list = []
for index, name in enumerate(classdir):
    classesdir = cwd + name + '/'
    for classes in os.listdir(classesdir):
        class_path = classesdir + classes + '/'
        list.append(len(os.listdir(class_path)))


# 生成tfrecord文件
def pro_tfrecords(data_type):
    imglist = []
    with tf.python_io.TFRecordWriter('/home/vbuo/m-L-1/' + data_type +
                                     '.tfrecords') as writer:
        for index, name in enumerate(classdir):
            classesdir = cwd + name + '/'
            for classes in os.listdir(classesdir):
                class_path = classesdir + classes + '/'
                for img_name in os.listdir(class_path):
                    # 获取此文件夹中文件地址，加入列表
                    img_path = class_path + img_name
                    imglist.append(img_path)
                # 训练集取75%，测试集取25%
                if data_type == 'train':
                    l = imglist[0:int(0.75 * min(list))]
                else:
                    l = imglist[int(0.75 * min(list)):min(list)]
                for i in l:
                    img = Image.open(i)
                    img1 = img.resize((24, 48))
                    # 转灰度图像
                    image = img1.convert("L")
                    img_raw = image.tobytes()
                    example = tf.train.Example(features=tf.train.Features(
                        feature={
                            "label":
                            tf.train.Feature(int64_list=tf.train.Int64List(
                                value=[Label[classes]])),
                            'img_raw':
                            tf.train.Feature(bytes_list=tf.train.BytesList(
                                value=[img_raw]))
                        }))
                    writer.write(example.SerializeToString())  #序列化为字符串
                imglist = []


pro_tfrecords('train')
pro_tfrecords('test')
print("success")
