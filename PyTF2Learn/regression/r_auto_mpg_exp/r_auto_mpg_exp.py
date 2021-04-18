# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 在线下载汽车效能数据集
dataset_path = keras.utils.get_file(
    "auto-mpg.data",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)

# 效能(公里数每加仑),气缸数,排量,马力,重量,加速度,型号年份,产地
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower',
                'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?",
                          comment='\t', sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
# 查看部分数据
dataset.tail()
dataset.head()
# dataset

# 统计空白数据,并清除
dataset.isna().sum()
dataset = dataset.dropna()
dataset.isna().sum()
# dataset

# 处理类别型数据，其中origin列代表了类别1,2,3,分布代表产地：美国、欧洲、日本
origin = dataset.pop('Origin')  # 其弹出这一列
# 根据origin列来写入新列
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0
dataset.tail()

# 切分为训练集和测试集
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# 统计数据
sns.pairplot(train_dataset[["Cylinders", "Displacement", "Weight", "MPG"]], diag_kind="kde")

# 查看训练集的输入X的统计数据
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
# train_stats

# 移动MPG油耗效能这一列为真实标签Y
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


# 标准化数据
def norm(data):
    return (data - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
print(normed_train_data.shape, train_labels.shape)
print(normed_test_data.shape, test_labels.shape)


class Network(keras.Model):  # 回归网络
    def __init__(self):
        super(Network, self).__init__()
        # 创建3个全连接层
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(1)

    # call(self, inputs, training=None, mask=None)
    def call(self, inputs):
        # 依次通过3个全连接层
        o = self.fc1(inputs)
        o = self.fc2(o)
        o = self.fc3(o)
        return o


model = Network()
model.build(input_shape=(None, 9))
model.summary()
optimizer = tf.keras.optimizers.RMSprop(0.001)
train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values, train_labels.values))
train_db = train_db.shuffle(100).batch(32)

# # 未训练时测试
# example_batch = normed_train_data[:10]
# example_result = model.predict(example_batch)
# example_result

train_mae_losses = []
test_mae_losses = []
for epoch in range(200):
    for step, (x, y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            out = model.call(x)
            train_loss = tf.reduce_mean(losses.MSE(y, out))
            # train_mae_loss = tf.reduce_mean(losses.MAE(y, out))
            train_mae_loss = tf.reduce_mean(losses.MAE(train_labels, out))
        if epoch % 10 == 0 and step % 10 == 0:
            print(epoch, step, float(train_loss))
        grads = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # train_mae and test_mae
    train_mae_losses.append(float(train_mae_loss))
    out = model.call(tf.constant(normed_test_data.values))
    test_mae_loss = tf.reduce_mean(losses.MAE(test_labels, out))
    test_mae_losses.append(float(test_mae_loss))

plt.figure()
plt.plot(train_mae_losses, label='Train')
plt.plot(test_mae_losses, label='Test')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()  # 添加图例
plt.savefig('r_auto_mpg_mae.svg')
plt.show()
exit()
