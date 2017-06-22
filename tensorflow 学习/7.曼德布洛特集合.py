# coding=utf-8
# mandelbrot 集合

# 导入仿真库
import tensorflow as tf
import numpy as np

# 导入可视化库
import PIL.Image
from cStringIO import StringIO
from IPython.display import clear_output, Image, display
import scipy.ndimage as nd
import matplotlib.pyplot as plt

# 定义一个函数来显示迭代计算出来的图像
def DisplayFractal(a, fmt='jpeg'):
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape) + [1])
    img = np.concatenate([10 + 20 * np.cos(a_cyclic),
                         30 + 50 * np.sin(a_cyclic),
                         155 - 80 * np.cos(a_cyclic)],
                         2
                         )
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

#  display 在jupter notebook中才能显示，这里暂用plt显示
    plt.imshow(PIL.Image.fromarray(a))
    plt.show()

# 会话和变量Variable初始化
sess = tf.InteractiveSession()
# 创建一个在【-2， 2】x 【-2， 2】范围内的2维复数数组
Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
Z = X + 1j*Y

# 初始化张量
xs = tf.constant(Z.astype('complex64'))
zs = tf.Variable(xs)
ns = tf.Variable(tf.zeros_like(xs, 'float32'))

tf.global_variables_initializer().run()

# --- 定义并运行计算 ---
# 计算一个新值
zs_ = zs*zs + xs

# 这个值会发散吗
not_diverged = tf.abs(zs_) < 4

# 更新zs并迭代计算
step = tf.group(
    zs.assign(zs_),
    ns.assign_add(tf.cast(not_diverged, 'float'))
)

for i in range(100):
    step.run()

DisplayFractal(ns.eval())

