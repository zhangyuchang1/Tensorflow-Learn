# coding=utf-8
# Python在默认状态下不支持源文件中的编码所致
'''
生成一些三维数据，然后用一个平面拟合他
'''
import tensorflow as tf
import numpy as np

# 使用numpy 生成假数据 ，共100个点
x_1 = np.random.rand(2, 100)  # 维度
x_data = np.float32(x_1)
# print x_1
# print x_data
y_data = np.dot([0.100, 0.200], x_data) + 0.300
# print y_data

# 构造一个线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 启动图 graph
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in xrange(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(W), sess.run(b)



