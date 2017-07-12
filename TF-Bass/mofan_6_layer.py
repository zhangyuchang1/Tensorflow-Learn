# 一层神经网络层 包含权重，偏置， 激活函数，
import tensorflow as tf

def add_layer(inputs, in_size, out_size, actiovation_function=None):

    # 权重矩阵  用随机变量会比0 好一点， 形状与输入相同 in_size行，out_size列
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))
    # 偏置 列表 初始值推荐不为0 所有+ 0.1
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.multiply(inputs, Weight) + biases

    if actiovation_function is None:
        outputs = Wx_plus_b # 说明是线性
    else:
        outputs = actiovation_function(Wx_plus_b)
    return outputs



