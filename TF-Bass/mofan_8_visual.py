# 训练数据与结果可视化
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 添加一层
def add_layer(inputs, in_size, out_size, actiovation_function=None):

    # 权重矩阵  用随机变量会比0 好一点， 形状与输入相同 in_size行，out_size列
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    # 偏置 列表 初始值推荐不为0 所有+ 0.1
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weight) + biases

    if actiovation_function is None:
        outputs = Wx_plus_b # 说明是线性
    else:
        outputs = actiovation_function(Wx_plus_b)
    return outputs


# 创建一些数据
# -1 到1有300个单位，再加一个列维度，也就是 500行 1列
x_data = np.linspace(-1,1,300)[:,np.newaxis]
# 添加一些噪声 标准正态分布的方法
noise = np.random.normal(0, 0.05,x_data.shape)
y_data = np.square(x_data) -0.5 + noise

# None表示无论多少个都ok
xs = tf.placeholder(dtype=tf.float32, shape=[None, 1])
ys = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# 3层 输入层1个  隐藏层10个神经元  输出层1个
l_1 = add_layer(xs, 1, 10, actiovation_function=tf.nn.relu)
predition = add_layer(l_1, 10, 1, actiovation_function=None)

# 平均的误差 损失函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition),
                     reduction_indices=[1]))

# 训练步骤就是 tf.train的一个损失优化过程
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 画图
fig = plt.figure()   # 图片框
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()


for step in range(1000):
    sess.run(train_step, feed_dict={xs: x_data,
                                    ys: y_data})
    if step % 50 == 0:
        loss_ = sess.run(loss, feed_dict={xs:x_data,
                                         ys:y_data})
        print(loss_)
        try:
            ax.lines.remove(lines[0])  # 去掉上一条线
        except Exception:
            pass

        prediction_value = sess.run(predition, feed_dict={xs:x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.5)  # 不要让plt 马上关闭




