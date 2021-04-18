# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential, regularizers


N_EPOCHS = 500
N_SAMPLES = 1000
TEST_SIZE = 0.3


def load_dataset():
    X, y = make_moons(n_samples=N_SAMPLES, noise=0.25, random_state=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    return X, y, X_train, y_train, X_test, y_test


def make_plot(X, y, plot_name, file_name, XX=None, YY=None, preds=None):
    plt.figure()
    axes = plt.gca()
    axes.set_xlim([-2, 3])
    axes.set_ylim([-1.5, 2])
    axes.set(xlabel='$x_1$', ylabel='$x_2$')
    plt.title(plot_name, fontsize=20, fontproperties='SimHei')
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    if XX is not None and YY is not None and preds is not None:
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha=0.08, cmap='Spectral')
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], cmap='Greys', vmin=0, vmax=.6)
    markers = ['o' if i == 1 else 's' for i in y.ravel()]
    mscatter(X[:, 0], X[:, 1], c=y.ravel(), s=20, cmap='Spectral', edgecolors='none', m=markers, ax=axes)
    plt.savefig(file_name)
    plt.show()
    plt.close()


def mscatter(x, y, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax:
        ax = plt.gca()
    sc = ax.scatter(x, y, **kw)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def network_layers_influence(X_train, y_train):
    for n in range(5):
        model = Sequential()
        model.add(Dense(8, input_dim=2, activation='relu'))
        for _ in range(n):
            model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=N_EPOCHS, verbose=1)
        xx = np.arange(-2, 3, 0.01)
        yy = np.arange(-1.5, 2, 0.01)
        XX, YY = np.meshgrid(xx, yy)
        preds = model.predict_classes(np.c_[XX.ravel(), YY.ravel()])
        title = '网络层数: {0}'.format(n + 2)
        file = '网络容量_%i.png' % (n + 2)
        make_plot(X_train, y_train, title, file, XX, YY, preds)


def dropout_influence(X_train, y_train):
    for n in range(5):
        model = Sequential()
        model.add(Dense(8, input_dim=2, activation='relu'))
        counter = 0
        for _ in range(5):
            model.add(Dense(64, activation='relu'))
            if counter < n:
                counter += 1
                model.add(Dropout(rate=0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=N_EPOCHS, verbose=1)
        xx = np.arange(-2, 3, 0.01)
        yy = np.arange(-1.5, 2, 0.01)
        XX, YY = np.meshgrid(xx, yy)
        preds = model.predict_classes(np.c_[XX.ravel(), YY.ravel()])
        title = '{0}层 Dropout层'.format(n)
        file = 'Dropout_%i.png' % n
        make_plot(X_train, y_train, title, file, XX, YY, preds)


def build_model_with_regularization(_lambda):
    model = Sequential()
    model.add(Dense(8, input_dim=2, activation='relu'))
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(_lambda)))
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(_lambda)))
    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(_lambda)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def plot_weights_matrix(model, layer_index, plot_name, file_name):
    weights = model.layers[layer_index].get_weights()[0]
    shape = weights.shape
    X = np.array(range(shape[1]))
    Y = np.array(range(shape[0]))
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    plt.title(plot_name, fontsize=20, fontproperties='SimHei')
    ax.plot_surface(X, Y, weights, cmap=plt.get_cmap('rainbow'), linewidth=0)
    ax.set_xlabel('网格x坐标', fontsize=16, rotation=0, fontproperties='SimHei')
    ax.set_ylabel('网格y坐标', fontsize=16, rotation=0, fontproperties='SimHei')
    ax.set_zlabel('权值', fontsize=16, rotation=90, fontproperties='SimHei')
    plt.savefig(file_name + '.svg')
    plt.close(fig)


def regularization_influence(X_train, y_train):
    for _lambda in [1e-5, 1e-3, 1e-1, 0.12, 0.13]:
        model = build_model_with_regularization(_lambda)
        model.fit(X_train, y_train, epochs=N_EPOCHS, verbose=1)
        layer_index = 2
        plot_title = "正则化系数：{}".format(_lambda)
        file_name = "正则化网络权值_" + str(_lambda)
        plot_weights_matrix(model, layer_index, plot_title, file_name)
        # 绘制不同正则化系数的决策边界线
        xx = np.arange(-2, 3, 0.01)
        yy = np.arange(-1.5, 2, 0.01)
        XX, YY = np.meshgrid(xx, yy)
        preds = model.predict_classes(np.c_[XX.ravel(), YY.ravel()])
        title = "正则化系数：{}".format(_lambda)
        file = "正则化系数_%g.png" % _lambda
        make_plot(X_train, y_train, title, file, XX, YY, preds)


if __name__ == '__main__':
    X, y, X_train, y_train, X_test, y_test = load_dataset()
    make_plot(X, y, 'Classification dataset visualization', 'dataset.svg')
    network_layers_influence(X_train, y_train)
    dropout_influence(X_train, y_train)
    regularization_influence(X_train, y_train)
    exit()
