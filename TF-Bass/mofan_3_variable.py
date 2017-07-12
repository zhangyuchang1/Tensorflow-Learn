# 变量

import tensorflow as tf

state = tf.Variable(1, name='counter')
print(state.name, '\n',state.value(),'\n' ,state.read_value())

one = tf.constant(1)

new_value = state + one

print('\n \n', new_value)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

print('\n \n', new_value)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 定义了变量一定要初始化所有变量激活

    for _ in range(3):
        sess.run(update)

        print('\n \n', state)
        print(sess.run(state))   # sess.run(state) 才能把Tensor 转会常量和通用数据
