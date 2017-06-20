# coding=utf-8
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

x = tf.placeholder('float', [None, 784]) # 表示一张图

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 实现模型
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 计算交叉熵 （损失函数）
y_ = tf.placeholder('float', [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 梯度下降法 以0.01的学习率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 开始训练
for _ in  range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})


# 评估模型
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})


# 约91%的正确率


