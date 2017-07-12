# 变量初始化为 placeholder，然后在 sess.run的时候 ，再feed 进值

import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    result = sess.run(output, feed_dict={input1: 3,
                                         input2: 5})
    print(result)