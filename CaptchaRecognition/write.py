import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

cwd = '/home/vbuo/m-L-2/save/images/test/'

filename_queue = tf.train.string_input_producer(
    ["/home/vbuo/m-L-2/save/tfrecord/test_1.tfrecords"])  #读入流中

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)  #返回文件名和文件
features = tf.parse_single_example(
    serialized_example,
    features={
        'label': tf.FixedLenFeature([], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string),
    })  #取出包含image和label的feature对象
# float16, float32, float64, int32, uint16, uint8, int16, int8, int64
image = tf.decode_raw(features['img_raw'], tf.uint8)
image = tf.reshape(image, [36, 48])
label = tf.cast(features['label'], tf.int64)
# print(label)
# print(image.shape)
with tf.Session() as sess:  #开始一个会话
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(4000):
        example, l = sess.run([image, label])  #在会话中取出image和label
        #example.show()
        data = np.matrix(example)
        # 变换成38*54
        data = np.reshape(data, (36, 48))
        img = Image.fromarray(data)
        img.save(cwd + str(i) + '_' 'Label_' + str(l) + '.jpg')  #存下图片
        print(example, l)
    coord.request_stop()
    coord.join(threads)
