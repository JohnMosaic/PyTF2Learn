# -*- encoding: UTF-8 -*-
import numpy as np


def random_sample(n):
    points = []
    for i in range(n):
        x = np.random.uniform(-10., 10.)
        eps = np.random.normal(0., 0.1)
        y = 1.477 * x + 0.089 + eps
        points.append([x, y])
    points = np.array(points)
    # print(points.shape, points)
    return points


def calc_mean_squared_error(w, b, points):
    total_error = 0
    n = len(points)
    for i in range(0, n):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (w * x + b - y) ** 2
    return total_error / float(n)


# learning_rate:
# 学习率大，学习速度快，适用于开始训练。缺点：易损失值爆炸，易振荡；
# 学习率小，学习速度慢，适用于训练一定轮数后。缺点：易过拟合，收敛速度慢
def step_gradient(w, b, points, learning_rate):
    w_gradient = 0
    b_gradient = 0
    n = len(points)
    for i in range(0, n):
        x = points[i, 0]
        y = points[i, 1]
        w_gradient += (w * x + b - y) * x * 2 / float(n)
        b_gradient += (w * x + b - y) * 2 / float(n)
    new_w = w - (learning_rate * w_gradient)
    new_b = b - (learning_rate * b_gradient)
    return [new_w, new_b]


def gradient_descent(init_w, init_b, points, learning_rate, iteration):
    w_gradient_descent = init_w
    b_gradient_descent = init_b
    for step in range(iteration):
        w_gradient_descent, b_gradient_descent = step_gradient(
            w_gradient_descent, b_gradient_descent, points, learning_rate)
        loss = calc_mean_squared_error(w_gradient_descent, b_gradient_descent, points)
        if (step <= 100 and step % 10 == 0) or (step > 100 and step % 100 == 0):
            print(f'iteration: {step}, w: {w_gradient_descent}, b: {b_gradient_descent}, loss: {loss}')
    return [w_gradient_descent, b_gradient_descent]


if __name__ == '__main__':
    arg_points = random_sample(100)
    arg_learning_rate = 0.001
    arg_w = 0
    arg_b = 0
    arg_iteration = 1000
    [final_w, final_b] = gradient_descent(arg_w, arg_b, arg_points, arg_learning_rate, arg_iteration)
    final_loss = calc_mean_squared_error(final_w, final_b, arg_points)
    print(f'Final w: {final_w}, b: {final_b}, loss: {final_loss}')
    print(f'Model: y = {final_w} * x + {final_b}')
    exit()
