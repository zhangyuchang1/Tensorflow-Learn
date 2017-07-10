import tensorflow as tf

# First conv
with tf.variable_scope('conv1'):
    w = tf.get_variable(name='weights',
                        shape=[3, 3, 3 ,16],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        trainable=False)
    b = tf.get_variable(name='biases',
                        shape=[16],
                        initializer=tf.constant_initializer(0.0),
                        trainable=False)
    print('w name:', w.name)
    print('b name: ', b.name)

with tf.name_scope('vonv1'):
    w = tf.get_variable(name='weights',
                        shape=[3, 3, 3, 16],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        trainable=False)
    b = tf.get_variable(name='biases',
                        shape=[16],
                        initializer=tf.constant_initializer(0.0),
                        trainable=False)
    print('w name:', w.name)
    print('b name: ', b.name)

# variable_scope 会自动加上前缀，用在初始化上
# name_scope 变量没前缀，用在给模型定义不同的节点时，放在一个name_scope下面画出tensorboard比较好看