import os
import csv
import tensorflow as tf
from PIL import Image

csvFile = open("label/labels.csv", "r")
reader = csv.reader(csvFile)  # 返回的是迭代类型
list = []
for item in reader:
    list.append(item[1])
print(len(list))
print(list[0])

# 数据所在文件夹
cwd = '/home/vbuo/m-L-2/images/'
cwd1 = '/home/vbuo/m-L-2/save/tfrecords/'
classes = os.listdir(cwd)
# 生成tfrecord文件每5000个数据分成4000训练500验证500测试 一共分8类
t_writer_1 = tf.python_io.TFRecordWriter(cwd1 + "train_1.tfrecords")  #生成全部文件
# v_writer_1 = tf.python_io.TFRecordWriter(
#     cwd1 + "validation_1.tfrecords")  #生成全部文件
# te_writer_1 = tf.python_io.TFRecordWriter(cwd1 + "test_1.tfrecords")  #生成全部文件

t_writer_2 = tf.python_io.TFRecordWriter(cwd1 + "train_2.tfrecords")  #生成全部文件
# v_writer_2 = tf.python_io.TFRecordWriter(
#     cwd1 + "validation_2.tfrecords")  #生成全部文件
# te_writer_2 = tf.python_io.TFRecordWriter(cwd1 + "test_2.tfrecords")  #生成全部文件

t_writer_3 = tf.python_io.TFRecordWriter(cwd1 + "train_3.tfrecords")  #生成全部文件
# v_writer_3 = tf.python_io.TFRecordWriter(
#     cwd1 + "validation_3.tfrecords")  #生成全部文件
# te_writer_3 = tf.python_io.TFRecordWriter(cwd1 + "test_3.tfrecords")  #生成全部文件

t_writer_4 = tf.python_io.TFRecordWriter(cwd1 + "train_4.tfrecords")  #生成全部文件
# v_writer_4 = tf.python_io.TFRecordWriter(
#     cwd1 + "validation_4.tfrecords")  #生成全部文件
# te_writer_4 = tf.python_io.TFRecordWriter(cwd1 + "test_4.tfrecords")  #生成全部文件

t_writer_5 = tf.python_io.TFRecordWriter(cwd1 + "train_5.tfrecords")  #生成全部文件
# v_writer_5 = tf.python_io.TFRecordWriter(
#     cwd1 + "validation_5.tfrecords")  #生成全部文件
# te_writer_5 = tf.python_io.TFRecordWriter(cwd1 + "test_5.tfrecords")  #生成全部文件

t_writer_6 = tf.python_io.TFRecordWriter(cwd1 + "train_6.tfrecords")  #生成全部文件
# v_writer_6 = tf.python_io.TFRecordWriter(
#     cwd1 + "validation_6.tfrecords")  #生成全部文件
# te_writer_6 = tf.python_io.TFRecordWriter(cwd1 + "test_6.tfrecords")  #生成全部文件

t_writer_7 = tf.python_io.TFRecordWriter(cwd1 + "train_7.tfrecords")  #生成全部文件
# v_writer_7 = tf.python_io.TFRecordWriter(
#     cwd1 + "validation_7.tfrecords")  #生成全部文件
# te_writer_7 = tf.python_io.TFRecordWriter(cwd1 + "test_7.tfrecords")  #生成全部文件

t_writer_8 = tf.python_io.TFRecordWriter(cwd1 + "train_8.tfrecords")  #生成全部文件
v_writer = tf.python_io.TFRecordWriter(
    cwd1 + "validation.tfrecords")  #生成全部文件
te_writer = tf.python_io.TFRecordWriter(cwd1 + "test.tfrecords")  #生成全部文件

for i in range(len(classes)):
    class_path = cwd + str(i) + '.jpg'
    img = Image.open(class_path)
    img1 = img.resize((48, 36))
    # 转灰度图像
    image = img.convert("L")
    print(i)
    img_raw = image.tobytes()
    example = tf.train.Example(features=tf.train.Features(
        feature={
            "label":
            tf.train.Feature(int64_list=tf.train.Int64List(
                value=[int(list[i])])),
            'img_raw':
            tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
    if i < 4000:
        # j = 1
        # t_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "train_" + str(j) + ".tfrecords")  #生成全部文件
        # v_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "validation_" + str(j) + ".tfrecords")  #生成全部文件
        # te_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "test_" + str(j) + ".tfrecords")  #生成全部文件
        # if i < 4000:
        t_writer_1.write(example.SerializeToString())  #序列化为字符串
        # elif i < 4500:
        #     v_writer_1.write(example.SerializeToString())  #序列化为字符串
        # else:
        #     te_writer_1.write(example.SerializeToString())  #序列化为字符串
    # t_writer_j.close()
    # v_writer_j.close()
    # te_writer_j.close()

    elif i < 8000:
        # j = 2
        # t_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "train_" + str(j) + ".tfrecords")  #生成全部文件
        # v_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "validation_" + str(j) + ".tfrecords")  #生成全部文件
        # te_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "test_" + str(j) + ".tfrecords")  #生成全部文件
        # if i < 9000:
        t_writer_2.write(example.SerializeToString())  #序列化为字符串
        # elif i < 9500:
        #     v_writer_2.write(example.SerializeToString())  #序列化为字符串
        # else:
        #     te_writer_2.write(example.SerializeToString())  #序列化为字符串
    # elif i < num
    # t_writer_j.close()
    # v_writer_j.close()
    # te_writer_j.close()
    elif i < 12000:
        # j = 3
        # t_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "train_" + str(j) + ".tfrecords")  #生成全部文件
        # v_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "validation_" + str(j) + ".tfrecords")  #生成全部文件
        # te_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "test_" + str(j) + ".tfrecords")  #生成全部文件
        # if i < 14000:
        t_writer_3.write(example.SerializeToString())  #序列化为字符串
        # elif i < 14500:
        #     v_writer_3.write(example.SerializeToString())  #序列化为字符串
        # else:
        #     te_writer_3.write(example.SerializeToString())  #序列化为字符串
    # elif i < num
    # t_writer_j.close()
    # v_writer_j.close()
    # te_writer_j.close()
    elif i < 16000:
        # j = 4
        # t_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "train_" + str(j) + ".tfrecords")  #生成全部文件
        # v_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "validation_" + str(j) + ".tfrecords")  #生成全部文件
        # te_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "test_" + str(j) + ".tfrecords")  #生成全部文件
        # if i < 19000:
        t_writer_4.write(example.SerializeToString())  #序列化为字符串
        # elif i < 19500:
        #     v_writer_4.write(example.SerializeToString())  #序列化为字符串
        # else:
        #     te_writer_4.write(example.SerializeToString())  #序列化为字符串
    # elif i < num
    # t_writer_j.close()
    # v_writer_j.close()
    # te_writer_j.close()
    elif i < 20000:
        # j = 5
        # t_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "train_" + str(j) + ".tfrecords")  #生成全部文件
        # v_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "validation_" + str(j) + ".tfrecords")  #生成全部文件
        # te_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "test_" + str(j) + ".tfrecords")  #生成全部文件
        # if i < 24000:
        t_writer_5.write(example.SerializeToString())  #序列化为字符串
        # elif i < 24500:
        #     v_writer_5.write(example.SerializeToString())  #序列化为字符串
        # else:
        #     te_writer_5.write(example.SerializeToString())  #序列化为字符串
    # elif i < num
    # t_writer_j.close()
    # v_writer_j.close()
    # te_writer_j.close()
    elif i < 24000:
        # j = 6
        # t_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "train_" + str(j) + ".tfrecords")  #生成全部文件
        # v_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "validation_" + str(j) + ".tfrecords")  #生成全部文件
        # te_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "test_" + str(j) + ".tfrecords")  #生成全部文件
        # if i < 29000:
        t_writer_6.write(example.SerializeToString())  #序列化为字符串
        # elif i < 29500:
        #     v_writer_6.write(example.SerializeToString())  #序列化为字符串
        # else:
        #     te_writer_6.write(example.SerializeToString())  #序列化为字符串
    # elif i < num
    # t_writer_j.close()
    # v_writer_j.close()
    # te_writer_j.close()
    elif i < 28000:
        # j = 7
        # t_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "train_" + str(j) + ".tfrecords")  #生成全部文件
        # v_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "validation_" + str(j) + ".tfrecords")  #生成全部文件
        # te_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "test_" + str(j) + ".tfrecords")  #生成全部文件
        # if i < 34000:
        t_writer_7.write(example.SerializeToString())  #序列化为字符串
        # elif i < 34500:
        #     v_writer_7.write(example.SerializeToString())  #序列化为字符串
        # else:
        #     te_writer_7.write(example.SerializeToString())  #序列化为字符串
    # elif i < num
    # t_writer_j.close()
    # v_writer_j.close()
    # te_writer_j.close()
    elif i < 32000:
        #j = 8
        # t_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "train_" + str(j) + ".tfrecords")  #生成全部文件
        # v_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "validation_" + str(j) + ".tfrecords")  #生成全部文件
        # te_writer_j = tf.python_io.TFRecordWriter(
        #     cwd1 + "test_" + str(j) + ".tfrecords")  #生成全部文件
        # if i < 39000:
        t_writer_8.write(example.SerializeToString())  #序列化为字符串
        # elif i < 39500:
        #     v_writer_8.write(example.SerializeToString())  #序列化为字符串
        # else:
        #     te_writer_8.write(example.SerializeToString())  #序列化为字符串
    elif i < 36000:
        v_writer.write(example.SerializeToString())  #序列化为字符串
    else:
        te_writer.write(example.SerializeToString())  #序列化为字符串
    # elif i < num
    # t_writer_j.close()
    # v_writer_j.close()
    # te_writer_j.close()
t_writer_1.close()
# v_writer_1.close()
# te_writer_1.close()
t_writer_2.close()
# v_writer_2.close()
# te_writer_2.close()
t_writer_3.close()
# v_writer_3.close()
# te_writer_3.close()
t_writer_4.close()
# v_writer_4.close()
# te_writer_4.close()
t_writer_5.close()
# v_writer_5.close()
# te_writer_5.close()
t_writer_6.close()
# v_writer_6.close()
# te_writer_6.close()
t_writer_7.close()
# v_writer_7.close()
# te_writer_7.close()
t_writer_8.close()
# v_writer_8.close()
# te_writer_8.close()
v_writer.close()
te_writer.close()
csvFile.close()
