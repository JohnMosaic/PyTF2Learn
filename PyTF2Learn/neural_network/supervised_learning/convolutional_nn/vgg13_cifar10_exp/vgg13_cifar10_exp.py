# -*- coding: utf-8 -*-
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False
import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2345)
EPOCHS = 50
INTERVAL = 100

conv_layers = [  # 5 units of conv + max pooling
    # unit 1
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    # unit 2
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    # unit 3
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    # unit 4
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
    # unit 5
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
]


def preprocess(x, y):
    # [0~1]
    x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = datasets.cifar10.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print(x.shape, y.shape, x_test.shape, y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000).map(preprocess).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(64)

sample = next(iter(train_db))
print('sample:', sample[0].shape, sample[1].shape, tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))


def train_test():
    accuracies, losses = [], []
    conv_net = Sequential(conv_layers)  # [b, 32, 32, 3] => [b, 1, 1, 512]
    fc_net = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(10, activation=None),
    ])
    conv_net.build(input_shape=[None, 32, 32, 3])
    fc_net.build(input_shape=[None, 512])
    conv_net.summary()
    fc_net.summary()
    optimizer = optimizers.Adam(lr=1e-4)

    # [1, 2] + [3, 4] => [1, 2, 3, 4]
    variables = conv_net.trainable_variables + fc_net.trainable_variables

    for epoch in range(EPOCHS):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = conv_net(x)  # [b, 32, 32, 3] => [b, 1, 1, 512]
                out = tf.reshape(out, [-1, 512])  # flatten, => [b, 512]
                logits = fc_net(out)  # [b, 512] => [b, 10]
                y_onehot = tf.one_hot(y, depth=10)  # [b] => [b, 10]
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))
            if step % INTERVAL == 0:
                print(epoch, step, 'loss:', float(loss))
                losses.append(float(loss))

        total_num = 0
        total_correct = 0
        for x, y in test_db:
            out = conv_net(x)
            out = tf.reshape(out, [-1, 512])
            logits = fc_net(out)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            total_num += x.shape[0]
            total_correct += int(correct)
        acc = total_correct / total_num
        print(epoch, 'acc:', acc)
        accuracies.append(acc)
    return accuracies, losses


def make_plot(accuracies, losses):
    train_x = [i * INTERVAL for i in range(len(losses))]
    plt.figure()
    plt.plot(train_x, losses, color='C0', marker='s', label='训练')
    plt.ylabel('Crossentropy')
    plt.xlabel('Step')
    plt.legend()
    plt.savefig('vgg13_cifar10_exp_train_loss.svg')
    plt.show()

    test_x = [i * EPOCHS for i in range(len(accuracies))]
    plt.figure()
    plt.plot(test_x, accuracies, color='C1', marker='s', label='测试')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('vgg13_cifar10_exp_test_accuracy.svg')
    plt.show()


if __name__ == '__main__':
    accuracies, losses = train_test()
    make_plot(accuracies, losses)
    exit()
