import tensorflow as tf
import numpy as np


def read_data(path):
    filename = tf.matching_files(path)
    filename_queue = tf.train.string_input_producer(filename)
    reader = tf.TFRecordReader()
    _, example = reader.read(filename_queue)
    features = tf.parse_single_example(
        example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    images = tf.decode_raw(features['image_raw'], tf.uint8)
    images = tf.reshape(images, [1728])
    labels = tf.cast(features['label'], tf.int64)
    return images / 255, labels


# 将图片shape转化为(?, 4, 10 )
def image_reshape(l):
    lab = np.zeros([len(l), 4, 11])
    for n in range(len(l)):
        for j, j_ in enumerate(str(l[n]) + ' ' * (4 - len(str(l[n])))):
            if j_ == ' ':
                lab[n][j][10] = 1
            else:
                lab[n][j][int(j_)] = 1
    return lab


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))


def init_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


x_ = tf.placeholder(tf.float32, [None, 1728])
y = tf.placeholder(tf.float32, [None, 4, 11])
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
x = tf.reshape(x_, [-1, 36, 48, 1], name='x')

# 第一层卷积
w1 = init_weights([3, 3, 1, 32])
b1 = init_bias([32])
conv1 = tf.nn.relu(
    tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME') +
    b1)
# 池化
pool1 = tf.nn.max_pool(
    conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
    padding='SAME')

# 第二层卷积
w2 = init_weights([3, 3, 32, 64])
b2 = init_bias([64])
conv2 = tf.nn.relu(
    tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding='SAME') +
    b2)
# 池化
pool2 = tf.nn.max_pool(
    conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
    padding='SAME')

# 第三层卷积
w3 = init_weights([3, 3, 64, 96])
b3 = init_bias([96])
conv3 = tf.nn.relu(
    tf.nn.conv2d(pool2, w3, strides=[1, 1, 1, 1], padding='SAME') +
    b3)
# 池化
pool3_ = tf.nn.max_pool(
    conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
    padding='SAME')

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

cost1 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=y_oo[:, 0], labels=y[:, 0]))
cost2 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=y_oo[:, 1], labels=y[:, 1]))
cost3 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=y_oo[:, 2], labels=y[:, 2]))
cost4 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=y_oo[:, 3], labels=y[:, 3]))
cost = tf.reduce_sum([cost1, cost2, cost3, cost4])
train_op = tf.train.AdamOptimizer().minimize(cost)
prediction = tf.equal(tf.argmax(y_oo, 2), tf.argmax(y, 2))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name='accuracy')

t_accuracy = 0
batch_size = 1

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    image, label = read_data('./save/tfrecords/test.tfrecords')
    images, labels = tf.train.shuffle_batch(
        [image, label],
        batch_size=200,
        num_threads=1,
        capacity=2000,
        min_after_dequeue=100)
    saver.restore(sess, './save/model.ckpt')

    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(coord=coord)
    img, ll = sess.run([images, labels])
    lab = image_reshape(ll)
    test_batch = zip(
        range(0, len(img), batch_size),
        range(batch_size, len(img) + 1, batch_size))
    for start, end in test_batch:

        acc, a, b = sess.run(
            [accuracy,
             tf.argmax(y_oo, 2),
             tf.argmax(y, 2)],
            feed_dict={x_: img[start:end],
                       y: lab[start:end],
                       keep_prob: 1.0})

        if acc == 1:
            t_accuracy += 1
        print('正确率：', acc, '预测值：', a, '正确值：', b)
    print('正确率：', t_accuracy / 4000)
    coord.request_stop()
    coord.join(thread)
