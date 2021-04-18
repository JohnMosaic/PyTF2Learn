# -*- coding: utf-8 -*-
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


N_SAMPLES = 2000
TEST_SIZE = 0.3
X, y = make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
print(X.shape, y.shape)


def make_plot(X, y, plot_name, file_name=None, XX=None, YY=None, preds=None, dark=False):
    if dark:
        plt.style.use('dark_background')
    else:
        sns.set_style('whitegrid')
    plt.figure(figsize=(16, 12))
    axes = plt.gca()
    axes.set(xlabel='$x_1$', ylabel='$x_2$')
    plt.title(plot_name, fontsize=20)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    if XX is not None and YY is not None and preds is not None:
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha=1, cmap='Spectral')
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], cmap='Greys', vmin=0, vmax=.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap='Spectral', edgecolors='none')
    plt.savefig(file_name)
    plt.show()
    plt.close()


class Layer:
    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):
        """
        :param n_input: 输入节点数
        :param n_neurons: 输出节点数
        :param activation: 激活函数类型
        :param weights: 权值张量，默认类内部生成
        :param bias: 偏置，默认类内部生成
        """
        self.weights = weights if weights is not None else np.random.randn(n_input, n_neurons) * np.sqrt(1 / n_neurons)
        self.bias = bias if bias is not None else np.random.rand(n_neurons) * 0.1
        self.activation = activation
        self.last_activation = None
        self.error = None  # 用于计算当前层的delta变量的中间变量
        self.delta = None  # 记录当前层的delta变量，用于计算梯度

    def activate(self, x):
        r = np.dot(x, self.weights) + self.bias  # x@w+b
        self.last_activation = self._apply_activation(r)
        return self.last_activation

    def _apply_activation(self, r):
        if self.activation is None:
            return r
        elif self.activation == 'relu':
            return np.maximum(r, 0)
        elif self.activation == 'tanh':
            return np.tanh(r)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        return r

    def apply_activation_derivative(self, r):
        if self.activation is None:
            return np.ones_like(r)
        elif self.activation == 'relu':
            grad = np.array(r, copy=True)
            grad[r > 0] = 1.
            grad[r <= 0] = 0.
            return grad
        elif self.activation == 'tanh':
            return 1 - r ** 2
        elif self.activation == 'sigmoid':
            return r * (1 - r)
        return r


class NeuralNetwork:
    def __init__(self):
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)

    def feed_forward(self, X):
        for layer in self._layers:
            X = layer.activate(X)
        return X

    def back_propagation(self, X, y, learning_rate):
        output = self.feed_forward(X)
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]
            if layer == self._layers[-1]:
                layer.error = y - output
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else:
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)
        for i in range(len(self._layers)):  # 循环更新权值
            layer = self._layers[i]
            o_i = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            layer.weights += layer.delta * o_i.T * learning_rate

    def train_model(self, X_train, X_test, y_train, y_test, learning_rate, max_epochs):
        y_onehot = np.zeros((y_train.shape[0], 2))
        y_onehot[np.arange(y_train.shape[0]), y_train] = 1
        mses = []
        for i in range(max_epochs):
            for j in range(len(X_train)):
                self.back_propagation(X_train[j], y_onehot[j], learning_rate)
            if i % 10 == 0:
                mse = np.mean(np.square(y_onehot - self.feed_forward(X_train)))
                mses.append(mse)
                print(f'Epoch: {i} MSE: {float(mse)}')
                # print(f'Accuracy: {self.accuracy(self.predict(X_test), y_test.flatten())}')
        return mses


if __name__ == '__main__':
    make_plot(X, y, 'Classification dataset visualization', 'moon_dataset.svg')
    nn = NeuralNetwork()
    nn.add_layer(Layer(2, 25, 'sigmoid'))
    nn.add_layer(Layer(25, 50, 'sigmoid'))
    nn.add_layer(Layer(50, 25, 'sigmoid'))
    nn.add_layer(Layer(25, 2, 'sigmoid'))
    losses = nn.train_model(X_train, X_test, y_train, y_test, 0.01, 1000)
    x_labels = [i * 10 for i in range(len(losses))]
    plt.figure()
    plt.title('MSE loss', fontsize=20)
    plt.plot(x_labels, losses)
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.savefig('bp_moon_mse.svg')
    plt.show()
    exit()
