# coding:utf-8
# 别人的实现 参考http://blog.5ibc.net/p/103458.html

import numpy as np
import tensorflow as tf
import struct
import pickle
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

class Data:
    def __init__(self):
        self.K = 10
        self.N = 60000
        self.M = 10000
        self.BATCHSIZE = 100
        self.reg_factor = 1e-3
        self.stepsize = 1e-2
        self.train_img_list = np.zeros((self.N, 28*28))
        self.train_label_list = np.zeros((self.N, 1))
        self.test_img_list = np.zeros((self.M, 28*28))
        self.test_label_list = np.zeros((self.M, 1))

        self.sess = tf.InteractiveSession()
        self.loss_list = []

        self.init_network()
        self.sess.run(tf.global_variables_initializer())
        self.train_data = np.append(self.train_img_list, self.train_label_list, axis=1)

    def GetOneHot(self, transfer_list):
        const_zero = np.zeros([transfer_list.shape[0], 10])
        for i in range(transfer_list.shape[0]):
            const_zero[i][int(transfer_list[i])] = 1

        return  const_zero

    def get_weights(self, shape):
        initial = tf.truncated_normal(shape=shape, stddev=0.1)
        return tf.Variable(initial)

    def get_bias(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def init_network(self):
        self.label = tf.placeholder("float", [None, self.K])
        self.input_layer = tf.placeholder('float', [None, 28*28])
        self.input = tf.reshape(self.input_layer, (-1, 28, 28, 1))

        w1 = self.get_weights([5, 5, 1, 32])
        b1 = self.get_bias([32])
        w2 = self.get_weights([5, 5, 32, 64])
        b2 = self.get_bias([64])
        w_fc1 = self.get_weights([7*7*64, 1024])
        b_fc1 = self.get_bias([1024])
        w_fc2 = self.get_weights([1024, 10])
        b_fc2 = self.get_bias([10])

        h1 = tf.nn.relu(
             tf.nn.conv2d(self.input, w1, strides=[1, 1, 1, 1], padding='SAME') + b1
        )
        p1 = tf.nn.max_pool(h1, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        h2 = tf.nn.relu(
            tf.nn.conv2d(p1, w2, strides=[1, 1, 1, 1], padding='SAME') + b2
        )
        p2 = tf.nn.max_pool(h2, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        p2_flatten = tf.reshape(p2, [-1, 7*7*64])

        conv1 = tf.nn.relu(tf.matmul(p2_flatten, w_fc1) + b_fc1)
        self.output = tf.nn.softmax(tf.matmul(conv1, w_fc2) +b_fc2)
        cross_entropy = -tf.reduce_sum(self.label * tf.log(self.output))
        self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    def predict(self):
        batch = mnist.train.next_batch(50)
        prediction_scores = self.output.eval(feed_dict={self.input_layer: batch[0]})
        prediction = tf.argmax(prediction_scores, 1)
        correct_prediction = tf.equal(prediction, tf.argmax(batch[1], 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        print 'accuracy:  ', accuracy.eval()

    def train(self):
        for i in range(100):
            batch = mnist.train.next_batch(50)
            self.optimizer.run(
                feed_dict={self.input_layer: batch[0], self.label: batch[1]}
            )
            print i
            self.predict()
        return

def main():
    data = Data()
    data.train()
    data.predict()

if __name__ == '__main__':
    main()

























