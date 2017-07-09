#coding=utf-8

# 读取二进制文件

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

data_dir = 'data/'

def read_cifar10(data_dir, is_train, batch_size, is_suffle):
    '''
    读取数据
    :param data_dir:
    :param is_tain:
    :param batch_size:
    :param is_suffle:
    :return: lbael 1d tensor tf.int32, image 4d tensor [batch_size, height, width, 3] tf.float32
    '''
    img_width = 32
    img_height = 32
    img_depth = 3
    label_bytes = 1
    img_bytes = img_height * img_width * img_depth

    # 每个数据的长度 1 + 3072

    with tf.name_scope('input'):
        if is_train:
            filenames = [os.path.join(data_dir, 'data_batch_%d.bin' %ii)
                         for ii in np.arange(1, 6)]
        else:
            filenames = [os.path.join(data_dir, 'test_batch.bin')]

        filename_queue = tf.train.string_input_producer(filenames)
        reader = tf.FixedLengthRecordReader(label_bytes + img_bytes)
        key, value = reader.read(filename_queue)
        record_bytes = tf.decode_raw(value, tf.uint8)

        label = tf.slice(record_bytes, [0], [label_bytes])
        label = tf.cast(label, tf.int32)

        image_raw = tf.slice(record_bytes, [label_bytes], [img_bytes])
        image_raw = tf.reshape(image_raw, [img_depth, img_height, img_width])
        image = tf.transpose(image_raw, (1 ,2 ,0)) # convert from D/H/W to H/W/D
        image = tf.cast(image, tf.float32)
        ####################################
        # 可做一些数据增强处理
        # 随机裁剪 翻转 对比度 亮度
        #        image = tf.random_crop(image, [24, 24, 3])# randomly crop the image size to 24 x 24
        #        image = tf.image.random_flip_left_right(image)
        #        image = tf.image.random_brightness(image, max_delta=63)
        #        image = tf.image.random_contrast(image,lower=0.2,upper=1.8)
        #######################################

        image = tf.image.per_image_standardization(image)

        if is_suffle: # 打乱
           images, label_batch = tf.train.shuffle_batch(
                                 [image, label],
                                 batch_size = batch_size,
                                 num_threads=16,
                                 capacity= 2000,
                                min_after_dequeue=1500)
        else:
            images, label_batch = tf.train.batch(
                                  [image, label],
                                  batch_size = batch_size,
                                  num_threads=16,
                                  capacity=2000)

        return images, tf.reshape(label_batch, [batch_size])

        #one-hot
        #  使用 spreate_softmax激活函数需要onehot处理，使用softmax不需要

        n_class = 10
        # label_batch = tf.one_hot(label_batch, depth=n_class)
        #
        # return images, tf.reshape(label_batch, [batch_size, n_class])

# 测试一下能不能取到图片

def test():
    data_dir = 'data/'
    BATCH_SIZE = 10
    image_batch, label_batch = read_cifar10(data_dir,is_train=True, batch_size=BATCH_SIZE,is_suffle=True)
    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop() and i<1:
            # while i < 1:

                img, label = sess.run([image_batch, label_batch])

                # just one batch
                for j in np.arange(BATCH_SIZE):
                    # print('label: %d' %label[j])  # 格式不对
                    print('label' , label[j])
                    plt.imshow(img[j, : , : ,:])
                    plt.show()
                i += 1
        except tf.errors.OutOfRangeError:
            print('done')
        finally:
            coord.request_stop()

        coord.join(threads)


# test()

# 获取一张图片
def get_one_image():

    image = 0
    return image















