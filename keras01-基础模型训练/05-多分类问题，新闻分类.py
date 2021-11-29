import keras
from keras.datasets import reuters

# 数据处理
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

print(train_data.shape)  # (8982,)
print(test_labels.shape)  # (2246,)
print(test_labels)  # [ 3 10  1 ...  3  3 24]
print(test_data[0])  # 1 代表文章开始    2 代表超出常用单词以外的单词
print('新闻类别数量：{}'.format(train_labels.max()))

import numpy as np


def vectorize_sequences(sequences, dimension=10000):
    """
    :param sequences:
    :param dimension:
    :return:
    """
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


x_train = vectorize_sequences(train_data)  # 训练数据
x_test = vectorize_sequences(test_data)  # 验证数据


def to_one_hot(labels, dimension=46):  # dimension=46是新闻共有45类,新闻标签中最大值是45
    results = np.zeros(len(labels), dimension)
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


# one_hot_train_labels = to_one_hot(train_labels)
# one_hot_test_labels = to_one_hot(test_labels)

from keras.utils.np_utils import to_categorical

# to_categorical 方法与to_one_hot 效果相同
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

print(one_hot_test_labels[0], one_hot_test_labels.shape)

# 模型创建
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

print(model.summary())

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

x_val = x_train[:1000]  # 测试数据
partial_x_train = x_train[1000:]  # 训练数据

y_val = one_hot_train_labels[:1000]  # 测试标签
partial_y_train = one_hot_train_labels[1000:]  # 训练标签

history = model.fit(partial_x_train, partial_y_train, epochs=5, batch_size=128, validation_data=(x_val, y_val))

model.save('多分类问题_epochs=5.h5')

import matplotlib.pyplot as plt
import seaborn as sns

print(history.history.keys())
loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']


epochs = range(1, len(loss) + 1)

sns.set(font='FangSong', style='whitegrid')  # 简单地加上一个sns.set() 就可以设置成seaborn格式了
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(epochs, loss, 'bo', label='Training loss')
ax1.plot(epochs, val_loss, 'b', label='Validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation loss')
ax1.legend()

ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(epochs, acc, 'bo', label='Training acc')
ax2.plot(epochs, val_acc, 'b', label='Validation acc')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Acc')
ax2.set_title('Training and Validation acc')
ax2.legend()

plt.show()

"""
测试集损失函数在第四、五次迭代时就不在下降，精度在第五次迭代时就不在上升，所以迭代次数为5时就可以了
"""
