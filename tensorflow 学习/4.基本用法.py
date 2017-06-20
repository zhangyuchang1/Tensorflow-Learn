# coding=utf-8

# graph  计算任务
# op     graph图中的 节点
# Session 执行 gragh
# tensor 数据
# Variable 变量，维护状态
# feed fetch 赋值 取值

# 构建图
import tensorflow as tf
'''
# 创建一个常量 op， 产生一个 1*2 的矩阵， 这个op被作为一个节点加到默认图中
# 构建器的返回值代表该常量op的返回值
matrix1 = tf.constant([[3., 3.]])

# 创建另外一个常量op， 产生一个 2*1的矩阵
matrix2 = tf.constant([[2.], [2.]])

# 创建一个矩阵乘法op
product = tf.matmul(matrix1, matrix2)

print matrix1
print matrix2
print product

# 在一个会话中启动图 ,必须启动图才会进行运算
# 启动默认图
sess = tf.Session()
result = sess.run(product)
print result

matrix2 = sess.run(matrix2)
print matrix2

# 任务完成，关闭会话
sess.close()

# 或使用with，自动关闭
with tf.Session() as sess:
    result = sess.run([product])
    print result

# 若使用多核GPU
with tf.Session() as  sess:
    with tf.device('/gpu:1'):
        matrix1 = tf.constant([[3, 3]])
        matrix2 = tf.constant([[2], [2]])
        # ...
sess.close()
'''

'''
注意运行下面的代码要把上面的注释掉，否者会报错tensorflow.python.framework.errors_impl.InvalidArgumentError: <exception str() failed>
'''

# 变量Variable  注意运行下面的代码要把上面的注释掉，否者会报错tensorflow.python.framework.errors_impl.InvalidArgumentError: <exception str() failed>
# 创建一个变量，初始化为标量0
state = tf.Variable(0, name='counter')

# 创建一个op， 使state 增加1
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 变量必须先初始化
init_op = tf.global_variables_initializer()

# 启动图，运行op
with tf.Session() as sess:
    sess.run(init_op)
    print sess.run(state)

    for _ in range(5):
        sess.run(update)
        print sess.run(state)

# fetch 取值
input1 = tf.constant(3)
input2 = tf.constant(2)
input3 = tf.constant(5)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)

with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print result

# feed 赋值
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
   print sess.run([output], feed_dict={input1: [7.], input2:[2.]})


