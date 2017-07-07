# 对训练好的模型进行 准确率验证

import tensorflow as tf
import numpy as np
import cifar10_input

def evaluate():
    with tf.Graph().as_default():

        log_dir = 'log/'
        test_dir = 'data/'

        n_test = 10000

        # read test data
        images, labels =cifar10_input.read_cifar10(test_dir,)
