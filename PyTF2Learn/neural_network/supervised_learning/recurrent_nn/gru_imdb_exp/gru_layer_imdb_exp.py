# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, Sequential

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 512  # 批量大小
total_words = 10000  # 词汇表大小N_vocab
max_review_len = 80  # 句子最大长度s，大于的句子部分将截断，小于的将填充
embedding_len = 100  # 词向量特征长度f

# 加载IMDB数据集，此处的数据采用数字编码，一个数字代表一个单词
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
print(x_train.shape, len(x_train[0]), y_train.shape)
print(x_test.shape, len(x_test[0]), y_test.shape)

# 数字编码表
word_index = keras.datasets.imdb.get_word_index()
word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# 翻转编码表
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


decode_review(x_train[8])

# x_train:[b, 80]
# x_test: [b, 80]
# 截断和填充句子，使得等长，此处长句子保留句子后面的部分，短句子在前面填充
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

# 构建数据集，打散，批量，并丢掉最后一个不够batch_size的batch
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batch_size, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batch_size, drop_remainder=True)
print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test shape:', x_test.shape)


class MyRNN(keras.Model):  # Layer方式构建多层网络
    def __init__(self, units):
        super(MyRNN, self).__init__()
        # 词向量编码 [b, 80] => [b, 80, 100]
        self.embedding = layers.Embedding(total_words, embedding_len, input_length=max_review_len)
        # 构建RNN
        self.rnn = Sequential(
            [layers.GRU(units, dropout=0.5, return_sequences=True),
             layers.GRU(units, dropout=0.5)]
        )
        # 构建分类网络，用于将LAYER的输出特征进行分类，2分类
        # [b, 80, 100] => [b, 64] => [b, 1]
        self.out_layer = Sequential([layers.Dense(32), layers.Dropout(rate=0.5), layers.ReLU(), layers.Dense(1)])

    def call(self, inputs, training=None):
        x = inputs  # [b, 80]
        # embedding: [b, 80] => [b, 80, 100]
        x = self.embedding(x)
        # rnn layer compute,[b, 80, 100] => [b, 64]
        x = self.rnn(x)
        # 末层最后一个输出作为分类网络的输入: [b, 64] => [b, 1]
        x = self.out_layer(x, training)
        # p(y is pos|x)
        prob = tf.sigmoid(x)
        return prob


def make_plot(train_history):
    train_acc = train_history.history['accuracy']
    val_acc = train_history.history['val_accuracy']
    train_loss = train_history.history['loss']
    val_loss = train_history.history['val_loss']
    epochs = range(len(train_acc))
    plt.plot(epochs, train_acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'g', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.axis('on')
    plt.savefig('gru_layer_imdb_accuracy.svg')
    plt.show()

    plt.figure()
    plt.plot(epochs, train_loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'g', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.axis('on')
    plt.savefig('gru_layer_imdb_loss.svg')
    plt.show()


def run():
    units = 32  # RNN状态向量长度f
    epochs = 10  # 训练epochs
    model = MyRNN(units)
    # 装配
    model.compile(optimizer=optimizers.Adam(0.001), loss=losses.BinaryCrossentropy(), metrics=['accuracy'])
    # 训练和验证
    train_history = model.fit(db_train, epochs=epochs, validation_data=db_test)
    # 测试
    model.evaluate(db_test)
    # 绘制
    make_plot(train_history)


if __name__ == '__main__':
    run()
    exit()
