#　-*- coding: utf-8 -*-

from tensorflow.python.framework import graph_util
import tensorflow as tf
import numpy as np
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('my_list', '/home/vbuo/m-L-1/save', """存放模型的目录""")

train_batch = 300  # 训练集每批次样本数
validate_batch = 100  # 验证集每批次样本数
iterations = 61  # 训练次数


# 设计一个类，用于读取tfrecord文件和对训练集验证集的批量处理
class File:
    def __init__(self, path):
        self.path = path

    def filenamequeue(self):
        filename_queue = tf.train.string_input_producer([self.path])
        return filename_queue

    def readtfrecord(self):
        reader = tf.TFRecordReader()
        _, example = reader.read(self.filenamequeue())  #返回文件名和文件
        features = tf.parse_single_example(
            example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })  #取出包含image和label的feature对象

        images = tf.decode_raw(features['image_raw'], tf.uint8)
        images = tf.reshape(images, [1152])
        labels = tf.cast(features['label'], tf.int64)
        return images / 255, labels

    # 用于对训练集和验证集进行批量处理
    def get_batch(self, batch_size):
        with tf.name_scope('get_batch'):
            image, label = self.readtfrecord()
            images, labels = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=2,
                capacity=1000 + 3 * batch_size,
                min_after_dequeue=100)

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                coord = tf.train.Coordinator()
                thread = tf.train.start_queue_runners(coord=coord)
                img = sess.run(images)
                _lab = sess.run(labels)

                coord.request_stop()
                coord.join(thread)
                lab = np.zeros([batch_size, 65], dtype=int)
                for i in range(batch_size):
                    lab[i][_lab[i]] = 1
        return img, lab


# 权值初始化
def weight_variable(shape):
    # 用正态分布来初始化权值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # 本例中用relu激活函数，所以用一个很小的正偏置较好
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 读取tfrecord文件
train_data = File('n_l_train.tfrecords')
validate_data = File('n_l_validation.tfrecords')
# 对验证样本进行批量处理
val_img, val_label = validate_data.get_batch(validate_batch)

with tf.Graph().as_default() as T:
    x_ = tf.placeholder(tf.float32, [None, 1152], name='image')
    y = tf.placeholder(tf.float32, [None, 65], 'label')

    # 把x转为卷积所需要的形式
    x = tf.reshape(x_, [-1, 48, 24, 1], name='x')

    # 第一层卷积: 5×5×1卷积核32个 [5，5，1，32],conv1.shape=[-1, 48, 24, 32],学习32种特征
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    conv1 = tf.nn.relu(
        tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') +
        b_conv1)

    # 第一个pooling 层[-1, 48, 24, 32]->[-1, 24, 12, 32]
    pool1 = tf.nn.max_pool(
        conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME')

    # 第二层卷积: 5×5×32卷积核64个 [5，5，32，64],conv2.shape=[-1, 24, 12, 64]
    w_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    conv2 = tf.nn.relu(
        tf.nn.conv2d(pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') +
        b_conv2)

    # 第二个pooling 层,[-1, 24, 12, 64]->[-1, 12, 6, 64]
    pool2 = tf.nn.max_pool(
        conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME')

    # 第三层卷积: 5×5×64卷积核96个 [5，5，64，96],conv3.shape=[-1, 12, 6, 96]
    w_conv3 = weight_variable([5, 5, 64, 96])
    b_conv3 = bias_variable([96])
    conv3 = tf.nn.relu(
        tf.nn.conv2d(pool2, w_conv3, strides=[1, 1, 1, 1], padding='SAME') +
        b_conv3)

    # 第二个pooling 层,[-1, 14, 6, 64]->[-1, 6, 3, 96]
    pool3_ = tf.nn.max_pool(
        conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME')

    # flatten层，[-1, 6, 3, 96]->[-1, 6*3*96],即每个样本得到一个6*3*96维的样本
    pool3 = tf.reshape(pool3_, [-1, 6 * 3 * 96])

    # 全连接层
    W_fc1 = weight_variable([6 * 3 * 96, 512])
    b_fc1 = bias_variable([512])
    h_fc1 = tf.nn.relu(tf.matmul(pool3, W_fc1) + b_fc1)

    # dropout: 输出的维度和h_fc1一样，只是随机部分值被值为零
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 输出层
    w_out = weight_variable([512, 65])
    b_out = bias_variable([65])
    y_out = tf.matmul(h_fc1_drop, w_out) + b_out

    # 1.损失函数
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=y_out, labels=y))

    # 2.优化函数：AdamOptimizer
    train_op = tf.train.AdamOptimizer().minimize(cost)

    # 3.预测准确结果统计
    z = tf.argmax(y_out, 1)
    q = tf.arg_max(y, 1)
    correct_prediction = tf.equal(z, q)
    accuracy = tf.reduce_mean(tf.cast(
        correct_prediction, tf.float32), name='accuracy')

    test_acc_sum = tf.Variable(0.0)
    batch_acc = tf.placeholder(tf.float32)
    new_test_acc_sum = tf.add(test_acc_sum, batch_acc)
    update = tf.assign(test_acc_sum, new_test_acc_sum)
    saver = tf.train.Saver(max_to_keep=2)

with tf.Session(graph=T) as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter(FLAGS.my_list, sess.graph)
    ckpt = tf.train.latest_checkpoint(FLAGS.my_list)
    step = 0
    if ckpt:
        check_point_path = '/home/vbuo/m-L-1/save'  # 保存好模型的文件路径
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=check_point_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    allaccuracy = 0
    for i in range(iterations):
        train_accuracy, loss = sess.run(
            [accuracy, cost],
            feed_dict={
                x_: train_data.get_batch(train_batch)[0],
                y: train_data.get_batch(train_batch)[1],
                keep_prob: 1.0
            })
        print("step %d, accuracy: %g" % (i+1, train_accuracy))
        train_op.run(feed_dict={
            x_: train_data.get_batch(train_batch)[0],
            y: train_data.get_batch(train_batch)[1],
            keep_prob: 0.5
        })
        allaccuracy = allaccuracy + train_accuracy
    allaccuracy = allaccuracy/(iterations-1)
    print("Overall accuracy：　%g" % allaccuracy)
    coord.request_stop()
    coord.join(threads)

    new_graph = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, output_node_names=['accuracy'])
    tf.train.write_graph(new_graph, '', 'graph.pb', as_text=False)