# in_top_k 函数

# 判断target 是否在top k 的预测之中
# 输出batch_size 大小的bool 数组，如果对目标类的预测在所有预测的top k 中，
# 条目 out[i]=True。注意 InTopK 和 TopK 的操作是不同的。
# 如果多个类有相同的预测值并且跨过了top-k 的边界，所有这些类都被认为是在 top k中

import tensorflow as tf
import numpy as np

input = tf.constant(np.random.rand(3,4))
k = 2
# 这个函数的作用是返回 input 中每行最大的 k 个数，并且返回它们所在位置的索引。
output = tf.nn.top_k(input, k)
with tf.Session() as sess:
    input_ =(sess.run(input))
    output_ = (sess.run(output))
    print(input_)
    print(output_)


constant = np.random.rand(3,4)
input = tf.constant(constant, tf.float32)
k = 2   #targets对应的索引是否在最大的前k(2)个数据中
# 这个函数的作用是返回一个布尔向量，说明目标值是否存在于预测值之中
output = tf.nn.in_top_k(input, [3,3,3], k)
with tf.Session() as sess:
    input_ =(sess.run(input))
    output_ = (sess.run(output))
    print(input_)
    print(output_)


A = [[0.8, 0.6, 0.3], [0.1, 0.6, 0.4]]
B = [1, 1]
out = tf.nn.in_top_k(A, B, 1)
with tf.Session() as sess:

    out = sess.run(out)
    print(out)
