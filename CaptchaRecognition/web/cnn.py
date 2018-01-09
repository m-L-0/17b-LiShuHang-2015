import tensorflow as tf
import numpy as np
from PIL import Image


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))


def init_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def reshape_image(img_path):
    a = Image.open(img_path).convert('L')
    img1 = np.array(a.resize((48, 36)))
    image = np.reshape(img1, [-1, 1728])
    return image


def cnn(img_path, result=None):
    img = reshape_image(img_path)
    x_ = tf.placeholder(tf.float32, [None, 1728])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    x = tf.reshape(x_, [-1, 36, 48, 1], name='x')

    # 第一层卷积
    w1 = init_weights([3, 3, 1, 32])
    b1 = init_bias([32])
    conv1 = tf.nn.relu(
        tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME') +
        b1)  # shape=(?, 36, 48, 32)
    # 池化
    pool1 = tf.nn.max_pool(
        conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME')  # # shape=(?, 18, 24, 32)

    # 第二层卷积
    w2 = init_weights([3, 3, 32, 64])
    b2 = init_bias([64])
    conv2 = tf.nn.relu(
        tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding='SAME') +
        b2)  # shape=(?, 18, 24, 64)
    # 池化
    pool2 = tf.nn.max_pool(
        conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME')  # shape=(?, 9, 12, 64)

    # 第三层卷积
    w3 = init_weights([3, 3, 64, 96])
    b3 = init_bias([96])
    conv3 = tf.nn.relu(
        tf.nn.conv2d(pool2, w3, strides=[1, 1, 1, 1], padding='SAME') +
        b3)  # shape=(?, 9, 12, 96)
    # 池化
    pool3_ = tf.nn.max_pool(
        conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME')  # shape=(?, 5, 6, 96)

    pool3 = tf.reshape(pool3_, [-1, 5 * 6 * 96])

    # 全连接层
    w4 = init_weights([5 * 6 * 96, 512])
    b4 = init_bias([512])
    h = tf.nn.relu(tf.matmul(pool3, w4) + b4)
    h = tf.nn.dropout(h, keep_prob)

    # 输出层 1
    w_o = init_weights([512, 11])
    b_o = init_bias([11])
    y_o = tf.matmul(h, w_o) + b_o

    # 输出层 2
    w_o1 = init_weights([512, 11])
    b_o1 = init_bias([11])
    y_o1 = tf.matmul(h, w_o1) + b_o1

    # 输出层 3
    w_o2 = init_weights([512, 11])
    b_o2 = init_bias([11])
    y_o2 = tf.matmul(h, w_o2) + b_o2

    # 输出层 4
    w_o3 = init_weights([512, 11])
    b_o3 = init_bias([11])
    y_o3 = tf.matmul(h, w_o3) + b_o3

    y_oo = tf.concat([y_o, y_o1, y_o2, y_o3], 1)
    y_oo = tf.reshape(y_oo, [-1, 4, 11])

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './static/model/model.ckpt')

        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(coord=coord)
        res = sess.run(tf.argmax(y_oo, 2), feed_dict={x_: img, keep_prob: 1.0})

        coord.request_stop()
        coord.join(thread)
        res = list(res[0])
        result = ''
        for i in res:
            if i == 10:
                pass
            else:
                result += str(i)
    return result
