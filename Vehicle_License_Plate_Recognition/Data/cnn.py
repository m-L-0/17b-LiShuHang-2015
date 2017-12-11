import tensorflow as tf
import os
import numpy as np
import random


# 设计一个类，用于读取tfrecord文件和对训练集验证集的批量处理
class File:
    # 读取tfrecord文件
    def readtfrecord(data_type='train', batch_size=10):
        filename_queue = tf.train.string_input_producer(
            ['/home/vbuo/m-L-1/' + data_type + '.tfrecords'])
        # 读取并解析一个tfrecord
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'img_raw': tf.FixedLenFeature([], tf.string),
            })
        image = tf.decode_raw(features['img_raw'], tf.uint8)
        image = tf.reshape(image, [1152])
        image = tf.cast(image, tf.float32) / 255
        label = tf.cast(features['label'], tf.int64)
        img_batch, l_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            capacity=500,
            min_after_dequeue=0)
        return img_batch, l_batch


    # 数据处理
    def read_tfrecord(data_type, num):
        img_b = np.empty([num, 1152])
        lab = np.empty([num, 34])
        img, l = File.readtfrecord(data_type, num)
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            img, l = sess.run([img, l])
            for i in range(num):
                img_b[i] = img[i]
                lab[i] = tf.one_hot(l[i], depth=34).eval()
                if i % 100 == 0:
                    print(i)
            coord.request_stop()
            coord.join(threads)
        return img_b, lab


#定义存储地址与名称
CNN = '/home/vbuo/m-L-1/save'
cnn = 'License_plate'


# 权值初始化
def weight_variable(shape):
    # 用正态分布来初始化权值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # 本例中用relu激活函数，所以用一个很小的正偏置较好
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial) 


def conv2d(x, W):
    #定义卷积层
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # 定义池化层
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# min_next_batch_tfr(随机批次载入数据)
def min_next_batch_tfr(image, label, num=50, num1=500):
    images = np.zeros((num, 1152))
    labels = np.zeros((num, 34))
    for i in range(num):
        temp = random.randint(0, num1 - 1)
        images[i, :] = image[temp]
        labels[i, :] = label[temp]

    return images, labels


x = tf.placeholder(tf.float32, [None, 1152])
y_ = tf.placeholder(tf.float32, [None, 34])
keep_prob = tf.placeholder("float")

# 第一层卷积: 5×5×1卷积核32个 [5，5，1，32],conv1.shape=[-1, 48, 24, 32],学习32种特征
W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])
# 格式转换
x_image = tf.reshape(x, [-1, 48, 24, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 第一个pooling 层[-1, 48, 24, 32]->[-1, 24, 12, 32]
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积: 5×5×32卷积核64个 [5，5，32，64],conv2.shape=[-1, 24, 12, 64]
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# 第二个pooling 层,[-1, 24, 12, 64]->[-1, 12, 6, 64]
h_pool2 = max_pool_2x2(h_conv2)

# 第三层卷积: 5×5×64卷积核96个 [5，5，64，96],conv3.shape=[-1, 12, 6, 96]
W_conv3 = weight_variable([3, 3, 64, 96])
b_conv3 = bias_variable([96])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
# 第三个pooling 层,[-1, 14, 6, 64]->[-1, 6, 3, 96]
h_pool3 = max_pool_2x2(h_conv3)

# flatten层，[-1, 6, 3, 96]->[-1, 6*3*96],即每个样本得到一个6*3*96维的样本
W_fc1 = weight_variable([6 * 3 * 96, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool3, [-1, 6 * 3 * 96])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# 使用drop out防止过拟合（正则化）
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层，输入1024维，输出34维，即0~33分类
W_fc2 = weight_variable([1024, 34])
b_fc2 = bias_variable([34])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
saver = tf.train.Saver(max_to_keep=1)

# 损失函数，交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
# 使用adam优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 准确度计算
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

x_train, y_train = File.read_tfrecord('train', 5000)
x_test, y_test = File.read_tfrecord('test', 1000)

# 启动多线程
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    j = 0
    losslist = []
    minloss = 100000
    # 若模型存在，自动加载模型进会话
    ckpt = tf.train.latest_checkpoint(CNN)
    if ckpt:
        saver.restore(sess=sess, save_path=ckpt)
        step = int(ckpt[len(os.path.join(CNN, cnn)) + 1:])
    ckptname = os.path.join(CNN, cnn)

    # 开启线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 训练
    for i in range(4000):
        batch = min_next_batch_tfr(x_train, y_train, 50, 5000)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        # 每一百次用验证集测试一次
        if i % 100 == 0:
            train_accuracy, loss = sess.run(
                [accuracy, cross_entropy],
                feed_dict={x: x_test,
                           y_: y_test,
                           keep_prob: 1})
            print("step %d, validating accuracy %g, loss is %g" %
                  (i, train_accuracy, loss))
            losslist.append(loss)
        # 保存损失最低的模型
        if minloss > loss:
            minloss = loss
            saver.save(sess, ckptname, global_step=i)
        if losslist[-1] > minloss and losslist[-2] > minloss and losslist[-3] > minloss:
            break

    # 损失函数列表
    print(losslist)
    print(minloss)
    coord.request_stop()
    coord.join(threads)

# # 验证数据集判断模型效果
# with tf.Session() as sess:
#     # 运行会话
#     sess.run(tf.global_variables_initializer())
#     step = 0
#     # 若模型存在，自动加载模型进会话
#     ckpt = tf.train.latest_checkpoint(CNN)
#     if ckpt:
#         saver.restore(sess=sess, save_path=ckpt)
#         step = int(ckpt[len(os.path.join(CNN, cnn)) + 1:])

#     # 开启线程
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#     # 测试
#     print("test accuracy %g" % accuracy.eval(
#         feed_dict={x: x_test,
#                    y_: y_test,
#                    keep_prob: 1.0}))
#     L_lmax = {}.fromkeys(range(34), 0)
#     L_lsame = {}.fromkeys(range(34), 0)
#     CL = {
#         0: '0',
#         1: '1',
#         2: '2',
#         3: '3',
#         4: '4',
#         5: '5',
#         6: '6',
#         7: '7',
#         8: '8',
#         9: '9',
#         10: 'A',
#         11: 'B',
#         12: 'C',
#         13: 'D',
#         14: 'E',
#         15: 'F',
#         16: 'G',
#         17: 'H',
#         18: 'J',
#         19: 'K',
#         20: 'L',
#         21: 'M',
#         22: 'N',
#         23: 'P',
#         24: 'Q',
#         25: 'R',
#         26: 'S',
#         27: 'T',
#         28: 'U',
#         29: 'V',
#         30: 'W',
#         31: 'X',
#         32: 'Y',
#         33: 'Z'
#     }
#     prediction_label = tf.argmax(
#         y_conv.eval(feed_dict={x: x_test,
#                                keep_prob: 1.0}), 1)
#     for i in range(len(x_test)):
#         if i % 100 == 0:
#             print(i)
#         L_lmax[int(np.argmax(y_test[i]))] += 1
#         if np.argmax(y_test[i]) == prediction_label[i].eval():
#             L_lsame[int(np.argmax(y_test[i]))] += 1
#     for i in range(34):
#         recall_rate = L_lsame[i] / L_lmax[i]
#         print("标签 %s 的召回率为 %f" % (CL[i], recall_rate))

#     coord.request_stop()
#     coord.join(threads)
