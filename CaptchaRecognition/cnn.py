import tensorflow as tf
import numpy as np


# 读取数据集
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
    print(images, labels)
    return images / 255, labels


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

# 第一层卷积: 3×3×1卷积核32个 [3，3，1，32],conv1.shape=[-1, 36, 48, 32],学习11种特征
w1 = init_weights([3, 3, 1, 32])
b1 = init_bias([32])
conv1 = tf.nn.relu(
    tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME') +
    b1)
# 第一个pooling 层[-1, 36, 48, 32]->[-1, 18, 24, 32]
pool1 = tf.nn.max_pool(
    conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
    padding='SAME')

# 第二层卷积: 3×3×32卷积核64个 [3，3，32，64],conv2.shape=[-1, 18, 24, 64]
w2 = init_weights([3, 3, 32, 64])
b2 = init_bias([64])
conv2 = tf.nn.relu(
    tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding='SAME') +
    b2)
# 第二个pooling 层[-1, 18, 24, 32]->[-1, 9, 12, 64]
pool2 = tf.nn.max_pool(
    conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
    padding='SAME')

# 第三层卷积: 3×3×64卷积核96个 [3，3，64，96],conv3.shape=[-1, 9, 12, 96]
w3 = init_weights([3, 3, 64, 96])
b3 = init_bias([96])
conv3 = tf.nn.relu(
    tf.nn.conv2d(pool2, w3, strides=[1, 1, 1, 1], padding='SAME') +
    b3)
# 第三个pooling 层[-1, 9, 12, 32]->[-1, 5, 6, 96]
pool3_ = tf.nn.max_pool(
    conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
    padding='SAME')

pool3 = tf.reshape(pool3_, [-1, 5 * 6 * 96])

# flatten层，[-1, 5, 6, 96]->[-1, 5*6*96],即每个样本得到一个5*6*96维的样本
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
# 损失函数
cost = tf.reduce_sum([cost1, cost2, cost3, cost4])
train_op = tf.train.AdamOptimizer().minimize(cost)
prediction = tf.equal(tf.argmax(y_oo, 2), tf.argmax(y, 2))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32), name='accuracy')

saver = tf.train.Saver()

batch_size = 100

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    image, label = read_data('./save/tfrecord/train*')
    images, labels = tf.train.shuffle_batch(
        [image, label],
        batch_size=32000,
        num_threads=1,
        capacity=2000,
        min_after_dequeue=1000)
    v_img, v_lab = read_data('./save/tfrecord/ver*')
    v_image, v_label = tf.train.shuffle_batch(
        [v_img, v_lab],
        batch_size=4000,
        num_threads=2,
        capacity=2000,
        min_after_dequeue=1000)

    for i in range(100):
        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(coord=coord)
        img, ll = sess.run([images, labels])
        v_images, v_labels = sess.run([v_image, v_label])
        lab = image_reshape(ll)
        v_lab = image_reshape(v_labels)
        training_batch = zip(
            range(0, len(img), batch_size),
            range(batch_size, len(img) + 1, batch_size))
        for start, end in training_batch:

            acc, co = sess.run(
                [accuracy, cost],
                feed_dict={
                    x_: img[start:end],
                    y: lab[start:end],
                    keep_prob: 1.0
                })

        print('训练次数：', i, '正确率：', acc, '损失率', co)
        if i % 5 == 0:
            v_accurary = sess.run(
                accuracy, feed_dict={x_: v_images,
                                     y: v_lab,
                                     keep_prob: 1.0})
            print('验证集：', v_accurary)

        if acc > 0.9 and v_accurary > 0.9:
            # sess.save(sess, "/home/vbuo/m-L-2/save/")
            save_path = saver.save(sess, "/home/vbuo/m-L-2/save/model.ckpt")
    coord.request_stop()
    coord.join(thread)
