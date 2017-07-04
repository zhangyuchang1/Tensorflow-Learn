# coding:utf-8
# 模型本身定义部分，确定出网络结构
# 输入层->卷积层->池化层->规范化层->卷积层->规范化层->池化层->全连接层->全连接层->softmax输出层

import tensorflow as tf
import input_dataset

#外部引用input_dataset文件中定义的hyperparameters
height = input_dataset.fixed_height
width = input_dataset.fixed_width
train_samples_per_epoch = input_dataset.train_samples_per_epoch
test_samples_per_epoch = input_dataset.test_sample_per_epoch

# 用于描述训练过程的常数
moving_average_decay = 0.9999    # the decay to use for the moving average
num_epoches_per_decay = 350.0    # 衰减呈阶梯函数， 控制衰减周期（阶梯宽度）
learning_rate_decay_factor = 0.1 # 学习衰减因子
initial_learning_rate = 0.1      # 初始学习率

def variable_on_cpu(name, shape, dtype, initializer):
    with tf.device('/cpu:0'): # 一个 context manager，用于为新的op指定要使用的硬件
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=initializer,
            dtype=dtype
        )
def variable_on_cpu_with_collection(name, shape, dtype, stddev, wd):
    with tf.device('/cpu:0'):
        weight = tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.truncated_normal_initializer( stddev=stddev, dtype=dtype)
        )
        if wd is None:
            weight_decay = tf.multiply(tf.nn.l2_loss(weight), wd, name='weight_loss')
            tf.add_to_collection(name='losses', value=weight_decay)

        return  weight

def losses_summary(total_loss):
# 通过使用指数衰减话，来维护变量的滑动均值，当训练模型时，维护训练参数的滑动均值有好处。
# 在测试过程中使用滑动参数比最终训练的参数本身
#  *会提高模型的正确率，apply（）方法会添加trained variables的shadow copies，并添加操作来维护变量的滑动均值到shadow copies
#  *方法可以访问shadow Variable，在创建evaluation model时非常有用
# 滑动均值是通过指数衰减计算得到的，shadow variable的初始化值和 trained variable相同，更新公式为：
# shadow_variable = decay * shadow_variable + (1 - decay) * variable

    average_op = tf.train.ExponentialMovingAverage(decay=0.9) # 创建一个新的指数滑动均值对象
    # 从字典集合中返回关键字'losses'对应的所有变量，包括交叉熵损失和正则项损失
    losses = tf.get_collection(key='losses')
    # 维护变量的滑动均值， 返回一个能够更新shadow Variables 的操作
    maintain_averages_op = average_op.apply(losses + [total_loss])
    for i in losses + [total_loss]:
        tf.summary.scalar(i.op.name + '_raw', 1)
        tf.summary.scalar(i.op.name, average_op.average(i))

    return maintain_averages_op 返回损失变量的更新操作

def one_step_train(total_loss, step):
    batch_count = int(train_samples_per_epoch/input_dataset.batch_size) # 训练块的个数
    decay_step = batch_count * num_epoches_per_decay # 每经过decay_step步训练，衰减lr
    lr = tf.train.exponential_decay(
         learning_rate=initial_learning_rate,
        global_step=step,
        decay_steps=decay_step,
        decay_rate=learning_rate_decay_factor,
        staircase=True
    )
    tf.summary.scalar('learning', lr)
    losses_movingaverage_op = losses_summary(total_loss)

    # control_dependencies是一个context manager，控制节点执行顺序，先执行control_inputs,再context
    with tf.control_dependencies(control_inputs=[losses_movingaverage_op]):
        trainer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        gradient_paris = trainer.compute_gradients(loss=total_loss)

    gradient_update = trainer.apply_gradients(grads_and_vars=gradient_pairs, global_step=step) #梯度更新



















































