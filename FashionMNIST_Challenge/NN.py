# -*- coding:utf-8
import numpy as np
import struct
import matplotlib.pyplot as plt
import random
import pickle

train_num = 1000


class Data:
    def __init__(self):

        self.K = 10
        self.N = 60000
        self.M = 10000
        self.BATCHSIZE = 2000
        self.reg_factor = 1e-3
        self.stepsize = 1e-2
        self.train_img_list = np.zeros((self.N, 28 * 28))
        self.train_label_list = np.zeros((self.N, 1))

        self.test_img_list = np.zeros((self.M, 28 * 28))
        self.test_label_list = np.zeros((self.M, 1))

        self.loss_list = []
        self.init_network()

        self.read_train_images('/home/vbuo/m-L-0/data/fashion/train-images-idx3-ubyte')
        self.read_train_labels('/home/vbuo/m-L-0/data/fashion/train-labels-idx1-ubyte')

        self.train_data = np.append(
            self.train_img_list, self.train_label_list, axis=1)

        self.read_test_images('/home/vbuo/m-L-0/data/fashion/t10k-images-idx3-ubyte')
        self.read_test_labels('/home/vbuo/m-L-0/data/fashion/t10k-labels-idx1-ubyte')

    def predict(self):
        hidden_layer1 = np.maximum(
            0, np.matmul(self.test_img_list, self.W1) + self.b1)

        hidden_layer2 = np.maximum(0,
                                   np.matmul(hidden_layer1, self.W2) + self.b2)

        scores = np.maximum(0, np.matmul(hidden_layer2, self.W3) + self.b3)

        prediction = np.argmax(scores, axis=1)
        prediction = np.reshape(prediction, (10000, 1))
        print(prediction.shape)
        print(self.test_label_list.shape)
        accuracy = np.mean(prediction == self.test_label_list)
        print('The accuracy is:  ', accuracy)
        return

    def train(self):

        for i in range(train_num):
            np.random.shuffle(self.train_data)
            img_list = self.train_data[:self.BATCHSIZE, :-1]
            label_list = self.train_data[:self.BATCHSIZE, -1:]
            print("Train Time: ", i)
            self.train_network(img_list, label_list)

    def train_network(self, img_batch_list, label_batch_list):

        # calculate softmax
        train_example_num = img_batch_list.shape[0]
        hidden_layer1 = np.maximum(
            0, np.matmul(img_batch_list, self.W1) + self.b1)

        hidden_layer2 = np.maximum(0,
                                   np.matmul(hidden_layer1, self.W2) + self.b2)

        scores = np.maximum(0, np.matmul(hidden_layer2, self.W3) + self.b3)

        scores_e = np.exp(scores)
        scores_e_sum = np.sum(scores_e, axis=1, keepdims=True)

        probs = scores_e / scores_e_sum

        loss_list_tmp = np.zeros((train_example_num, 1))
        for i in range(train_example_num):
            loss_list_tmp[i] = scores_e[i][int(label_batch_list[
                i])] / scores_e_sum[i]
        loss_list = -np.log(loss_list_tmp)

        loss = np.mean(loss_list, axis=0)[0] + 0.5 * self.reg_factor * np.sum(self.W1 * self.W1) + 0.5 * self.reg_factor * np.sum(self.W2 * self.W2) + 0.5 * self.reg_factor * np.sum(self.W3 * self.W3)

        self.loss_list.append(loss)
        print(loss, " ", len(self.loss_list))
        # backpropagation

        dscore = np.zeros((train_example_num, self.K))
        for i in range(train_example_num):
            dscore[i][:] = probs[i][:]
            dscore[i][int(label_batch_list[i])] -= 1

        dscore /= train_example_num

        dW3 = np.dot(hidden_layer2.T, dscore)
        db3 = np.sum(dscore, axis=0, keepdims=True)

        dh2 = np.dot(dscore, self.W3.T)
        dh2[hidden_layer2 <= 0] = 0

        dW2 = np.dot(hidden_layer1.T, dh2)
        db2 = np.sum(dh2, axis=0, keepdims=True)

        dh1 = np.dot(dh2, self.W2.T)
        dh1[hidden_layer1 <= 0] = 0

        dW1 = np.dot(img_batch_list.T, dh1)
        db1 = np.sum(dh1, axis=0, keepdims=True)

        dW3 += self.reg_factor * self.W3
        dW2 += self.reg_factor * self.W2
        dW1 += self.reg_factor * self.W1

        self.W3 += -self.stepsize * dW3
        self.W2 += -self.stepsize * dW2
        self.W1 += -self.stepsize * dW1

        self.b3 += -self.stepsize * db3
        self.b2 += -self.stepsize * db2
        self.b1 += -self.stepsize * db1

        return

    def init_network(self):
        self.W1 = 0.01 * np.random.randn(28 * 28, 100)
        self.b1 = 0.01 * np.random.randn(1, 100)

        self.W2 = 0.01 * np.random.randn(100, 20)
        self.b2 = 0.01 * np.random.randn(1, 20)

        self.W3 = 0.01 * np.random.randn(20, self.K)
        self.b3 = 0.01 * np.random.randn(1, self.K)

    def read_train_images(self, filename):
        binfile = open(filename, 'rb')
        buf = binfile.read()
        index = 0
        magic, self.train_img_num, self.numRows, self.numColums = struct.unpack_from(
            '>IIII', buf, index)
        print(magic, ' ', self.train_img_num, ' ', self.numRows, ' ', self.numColums)
        index += struct.calcsize('>IIII')
        for i in range(self.train_img_num):
            im = struct.unpack_from('>784B', buf, index)
            index += struct.calcsize('>784B')
            im = np.array(im)
            im = im.reshape(1, 28 * 28)
            self.train_img_list[i, :] = im

            # plt.imshow(im, cmap='binary')  # 黑白显示
            # plt.show()

    def read_train_labels(self, filename):
        binfile = open(filename, 'rb')
        index = 0
        buf = binfile.read()
        binfile.close()

        magic, self.train_label_num = struct.unpack_from('>II', buf, index)
        index += struct.calcsize('>II')

        for i in range(self.train_label_num):
            # for x in xrange(2000):
            label_item = int(struct.unpack_from('>B', buf, index)[0])
            self.train_label_list[i, :] = label_item
            index += struct.calcsize('>B')

    def read_test_images(self, filename):
        binfile = open(filename, 'rb')
        buf = binfile.read()
        index = 0
        magic, self.test_img_num, self.numRows, self.numColums = struct.unpack_from(
            '>IIII', buf, index)
        print(magic, ' ', self.test_img_num, ' ', self.numRows, ' ', self.numColums)
        index += struct.calcsize('>IIII')
        for i in range(self.test_img_num):
            im = struct.unpack_from('>784B', buf, index)
            index += struct.calcsize('>784B')
            im = np.array(im)
            im = im.reshape(1, 28 * 28)
            self.test_img_list[i, :] = im

    def read_test_labels(self, filename):
        binfile = open(filename, 'rb')
        index = 0
        buf = binfile.read()
        binfile.close()

        magic, self.test_label_num = struct.unpack_from('>II', buf, index)
        index += struct.calcsize('>II')

        for i in range(self.test_label_num):
            # for x in xrange(2000):
            label_item = int(struct.unpack_from('>B', buf, index)[0])
            self.test_label_list[i, :] = label_item
            index += struct.calcsize('>B')


def main():
    data = Data()
    data.train()
    data.predict()


if __name__ == '__main__':
    main()