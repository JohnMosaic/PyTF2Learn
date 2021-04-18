# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def himmelblau(arg_x):  # himmelblau函数实现
    return (arg_x[0] ** 2 + arg_x[1] - 11) ** 2 + (arg_x[0] + arg_x[1] ** 2 - 7) ** 2


x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x,y range:', x.shape, y.shape)
X, Y = np.meshgrid(x, y)  # 生成x-y平面采样网格点，方便可视化
print('X,Y maps:', X.shape, Y.shape)
Z = himmelblau([X, Y])  # 计算网格点上的函数值

# 绘制himmelblau函数曲面
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, cmap='gist_stern')
ax.view_init(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.savefig('himmelblau_3d.svg')
plt.show()

fig = plt.figure()
plt.contour(X, Y, Z, 200)
plt.ylabel('y')
plt.xlabel('x')
plt.savefig('himmelblau_contour.svg')
plt.show()

# 参数的初始化值对优化的影响不容忽视，可以通过尝试不同的初始化值，
# 检验函数优化的极小值情况
# [1., 0.], [-4, 0.], [4, 0.]
# x = tf.constant([4., 0.])
# x = tf.constant([1., 0.])
# x = tf.constant([-4., 0.])
x = tf.constant([-2., 2.])

for step in range(200):
    with tf.GradientTape() as tape:  # 梯度跟踪
        tape.watch([x])
        y = himmelblau(x)  # 前向传播
    grads = tape.gradient(y, [x])[0]  # 反向传播
    x -= 0.01 * grads  # 更新参数 0.01为学习率
    if step % 20 == 19:  # 打印优化的极小值
        print('step {}: x = {}, f(x) = {}'.format(step, x.numpy(), y.numpy()))

exit()
