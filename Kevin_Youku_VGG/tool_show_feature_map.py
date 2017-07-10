# 看神经网络提取的特征
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tools

def show_feature_map():
    cat = plt.imread('cat.jpg') # unit8
    plt.imshow(cat)    #[173, 237, 3]
    cat = tf.cast(cat, tf.float32)
    x = tf.reshape(cat, [1, 173, 237, 3])

    out = 25

    with tf.variable_scope('conv1') as scope:
        w = tools.weight([3, 3, 3, out], is_uniform=True)
        x_w = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        b = tools.bias([out])
        x_b = tf.nn.bias_add(x_w, b)
        x_relu = tf.nn.relu(x_b)

    n_feature = int(x_w.get_shape()[-1])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    feature_map = tf.reshape(x_w, [173, 237, out])
    images = tf.image.convert_image_dtype(feature_map, dtype=tf.uint8)
    images = sess.run(images)

    plt.figure(figsize=(10, 10))
    for i in np.arange(0, n_feature):
        plt.subplot(5, 5, i+1)
        plt.axis('off')
        plt.imshow(images[:,:,i])
    plt.show()


def show_rech_feature():

    cat = plt.imread('cat.jpg')  # unit8
    plt.imshow(cat)  # [173, 237, 3]
    cat = tf.cast(cat, tf.float32)
    x = tf.reshape(cat, [1, 173, 237, 3])

    with tf.variable_scope('conv1_1') as scope:

        w1 = tf.get_variable('wights', [3, 3, 3, 64])
        b1 = tf.get_variable('biases', [64])

        x_w = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
        x_b = tf.nn.bias_add(x_w, b1)
        x_relu = tf.nn.relu(x_b)

        out = 64

    n_feature = int(x_w.get_shape()[-1])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    feature_map = tf.reshape(x_relu, [173, 237, out])
    images = tf.image.convert_image_dtype(feature_map, dtype=tf.uint8)
    images = sess.run(images)

    plt.figure(figsize=(10, 10))
    for i in np.arange(0, 25):
        plt.subplot(5, 5, i + 1)
        plt.axis('off')
        plt.imshow(images[:, :, i])
    plt.show()



# show_feature_map()
show_rech_feature()


































