# coding=utf-8
# 数据读取 生成批次
# cat 0,dog 1 ,二分类

import tensorflow as tf
import numpy as np
import os


# 统一裁剪图片大小
img_width  = 208
img_height = 208

# train_file = 'data/train/'


def get_file(file_dir):
    '''
    :param file_dir: 图片文件路径
    :return: list of images and labels
    '''
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []

    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)

    print('there are %d cats\nthere are %d dogs' %(len(cats), len(dogs)))

    # 测试下
    # cats = cats[:5]
    # dogs = dogs[:5]
    # label_cats = label_cats[0:5]
    # label_dogs = label_dogs[0:5]
    #
    # print(cats)
    # print(dogs)

    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))

    #随机打乱， 这里打乱，生成批次的时候就不用打乱了
    temp = np.array([image_list, label_list])
    # print(temp)
    temp = np.transpose(temp)
    # print(temp)
    np.random.shuffle(temp)
    # print(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])

    # print(image_list)
    # print(label_list)
    # 字符串转成int
    label_list = [int(i) for i in label_list]

    return image_list, label_list

# 生成批次
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    
    :param image: list
    :param label: list
    :param image_W: width
    :param image_H: height
    :param batch_size: batch size
    :param capacity: queue中最多容纳的个数
    :return: image_batch:4D tensor [batch_size,width, height, 3] dtype=tf.float32
             label_batch:2D tensor [batch_size,]   dtype = tf.int32     
       
    '''

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make a input queue 输入队列
    input_queue = tf.train.slice_input_producer([image, label])

    # 从队里中取元素
    label = input_queue[1]
    # 图片解码
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # 这里要转下格式 视频上没有，好像不用
    # image = tf.cast(image,tf.float32)
    ##############################
    # 这里可以做一些 特征加强等等处理
    #############################

    # 图片裁剪和填充
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # 标准化处理
    image = tf.image.per_image_standardization(image)

    #生成批次 num_threads 线程数，酌情PC性能 capacity队列中最多容纳元素的个数
    image_batch, label_batch = tf.train.batch([image, label],batch_size,num_threads=36,capacity=capacity)
    print(label_batch)
    label_batch = tf.reshape(label_batch, [batch_size])
    print(label_batch)

    return  image_batch, label_batch



# 数据送到模型之前 可对数据测试一下
'''

import matplotlib.pyplot as plt

BATCH_SIZE  = 10
CAPACITY    = 256
IMG_H = 208
IMG_W = 208

train_file = 'data/train/'
image_list, label_list = get_file(train_file)
image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

with tf.Session() as sess:

    # 控制跑一两步
    i = 0
    # 监控状态， 入列 出列
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop() and i<1:
            img, label = sess.run([image_batch, label_batch])

            #just test one batch
            for j in np.arange(BATCH_SIZE):
                print('label: %d' %label[j])
                # print(img)
                # print(img[j, :, : ,:])
                # !!!!!!  为什么图片是 这个 ！！！！！！！
                plt.imshow(img[j, :, : ,:]) #4d 的
                plt.show()
            i +=1

    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()

    coord.join(threads)

'''


