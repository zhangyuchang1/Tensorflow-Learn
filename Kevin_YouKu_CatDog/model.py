# coding=utf-8
# 神经网络的结构

import  tensorflow as tf


def inference(images, batch_size, n_class):
    '''

    :param images: 4d tensor tf.float32 [batch_size, w, h, chanels]
    :param batch_size:
    :param n_class: 几分类
    :return: [batch_size, n_classes]
    '''

    # conv1 第一层卷积层
    # 这样tensorboard比较好看
    with tf.variable_scope('conv1') as scope:
        # shape = [kernel size 卷积核大小,kernel size 卷积核大小，channels通道，kernel numbers卷积核数量]
        weights = tf.get_variable('weights',
                                 shape=[3, 3, 3, 16],
                                 dtype=tf.float32,
                                 # 0.1 参考cifar10
                                 initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32)
                                 )
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 # 0.1 参考cifar10
                                 initializer=tf.constant_initializer(0.1)
                                 )
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # pool1 和 norm1
    with tf.variable_scope('pooling1_lrn') as scope:
        # ksize strides 数值是参考最常见的
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weight',
                                  shape=[3, 3, 16, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32)
                                  )
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

    # pool2 & norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                               padding='SAME', name='pooling2')
    # local3 全链接层
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32)
                                  )
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1)
                                 )

        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    #local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weight',
                                  shape=[128,128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32)
                                  )
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1)
                                 )
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    # softmax 层
    with tf.variable_scope('softmax_linear') as scope:
         weights = tf.get_variable('softmax_linear',
                                   shape=[128,n_class],
                                   dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32)
                                   )
         biases = tf.get_variable('biases',
                                  shape=[n_class],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1)
                                  )
         softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
         # 这里没有用激活函数，因为后面loss方法里面包含了激活函数

    return softmax_linear


# 损失函数
def losses(logits, labels):
    '''
    
    :param logits: logits tensor [batch_size, n_classes]
    :param label: label tensor [batch_size]
    :return:  loss tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        # 用sparse_softmax_cross_entropy_with_logits 就不用one-hot encoding了，用softmax需要
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        # tensorboard 显示loss变化
        tf.summary.scalar(scope.name+'/loss',loss)

    return  loss

# 训练
def training(loss, learning_rate):

    with tf.name_scope('optimizer'):
        # 优化器
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)

    return  train_op

# 评估准确率
def evaluation(logits, labels):
    '''
    
    :param logits: logits tensor, float - [batch_size, NUM_CLASS]
    :param labels: labels tensor, in32 - [batch_size]
    :return: a scalar int32
    '''
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        # 取均值
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy', accuracy)
    return accuracy
















