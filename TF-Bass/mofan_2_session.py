import tensorflow as tf

# matrix1 = tf.Variable([[3, 3]], dtype=tf.int32)  # shape = （1， 2）
matrix1 = tf.constant([[3, 3]], dtype=tf.int32)  # shape = （1， 2）

matrix2 = tf.constant([[2],      # shape = （2， 1）
                       [2]])
product = tf.matmul(matrix1, matrix2)

sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# 第二种方法带with
with tf.Session() as sess:

    result = sess.run(product)
    print(result)
