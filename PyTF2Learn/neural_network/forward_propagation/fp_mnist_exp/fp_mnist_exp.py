# -*- coding: utf-8 -*-
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['figure.titlesize'] = 20
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['font.family'] = ['STKaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False
import tensorflow as tf
from tensorflow.keras import datasets
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def preprocess(arg_x, arg_y):
    print(arg_x.shape, arg_y.shape)  # [b, 28, 28], [b]
    arg_x = tf.cast(arg_x, dtype=tf.float32) / 255.
    arg_x = tf.reshape(arg_x, [-1, 28 * 28])
    arg_y = tf.cast(arg_y, dtype=tf.int32)
    arg_y = tf.one_hot(arg_y, depth=10)
    return arg_x, arg_y


(x, y), (x_test, y_test) = datasets.mnist.load_data()
print('x:', x.shape, 'y:', y.shape, 'x test:', x_test.shape, 'y test:', y_test)
batch_size = 512
train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(1000)
train_db = train_db.batch(batch_size)
train_db = train_db.map(preprocess)
train_db = train_db.repeat(20)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.shuffle(1000).batch(batch_size).map(preprocess)
x, y = next(iter(train_db))
print('train sample:', x.shape, y.shape)
# print(x[0], y[0])


def run_model():
    lr = 1e-2  # learning rate
    accuracies, losses = [], []
    # 784 => 512
    w1, b1 = tf.Variable(tf.random.normal([784, 256], stddev=0.1)), tf.Variable(tf.zeros([256]))
    # 512 => 256
    w2, b2 = tf.Variable(tf.random.normal([256, 128], stddev=0.1)), tf.Variable(tf.zeros([128]))
    # 256 => 10
    w3, b3 = tf.Variable(tf.random.normal([128, 10], stddev=0.1)), tf.Variable(tf.zeros([10]))

    for step, (train_x, train_y) in enumerate(train_db):
        train_x = tf.reshape(train_x, (-1, 784))  # [b, 28, 28] => [b, 784]
        with tf.GradientTape() as tape:
            h1 = train_x@w1 + b1  # layer1
            h1 = tf.nn.relu(h1)
            h2 = h1@w2 + b2  # layer2
            h2 = tf.nn.relu(h2)
            out = h2@w3 + b3  # output

            # [b, 10] - [b, 10]
            loss = tf.square(train_y - out)  # compute loss
            # [b, 10] => scalar
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        for p, g in zip([w1, b1, w2, b2, w3, b3], grads):
            p.assign_sub(lr * g)

        if step % 80 == 0:
            print(f'step: {step} loss: {float(loss)}')
            losses.append(float(loss))

        if step % 80 == 0:
            total, total_correct = 0., 0
            for test_x, test_y in test_db:  # evaluate/test
                h1 = test_x@w1 + b1  # layer1
                h1 = tf.nn.relu(h1)
                h2 = h1@w2 + b2  # layer2
                h2 = tf.nn.relu(h2)
                out = h2@w3 + b3  # output
                pred = tf.argmax(out, axis=1)  # [b, 10] => [b]
                test_y = tf.argmax(test_y, axis=1)  # convert one_hot y to number y
                correct = tf.equal(pred, test_y)  # bool type
                # bool tensor => int tensor => numpy
                total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
                total += test_x.shape[0]
            print(f'Step: {step} Evaluate Acc: {total_correct / total}')
            accuracies.append(total_correct / total)

    plt.figure()
    train_x = [i * 80 for i in range(len(losses))]
    plt.plot(train_x, losses, color='C0', marker='s', label='训练')
    plt.ylabel('MSE')
    plt.xlabel('Step')
    plt.legend()
    plt.savefig('fp_mnist_train_mse.svg')
    plt.show()

    plt.figure()
    plt.plot(train_x, accuracies, color='C1', marker='s', label='测试')
    plt.ylabel('Accuracy')
    plt.xlabel('Step')
    plt.legend()
    plt.savefig('fp_mnist_test_accuracy.svg')
    plt.show()


if __name__ == '__main__':
    run_model()
    exit()
