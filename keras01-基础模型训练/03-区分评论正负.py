from keras.datasets import imdb
import numpy as np

# 获取imdb数据,取最常见的10000个单词
from matplotlib.axes import Axes

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(train_data[0])
print(train_labels[0])
print(train_labels)
print('train_labels:{}'.format(train_labels.shape))
print('train_data:{}'.format(train_data.shape))
print('test_data:{}'.format(test_data.shape))
print('test_labels:{}'.format(test_labels.shape))

print(max([max(sequence) for sequence in train_data]))  # train_data 中最大的数字


# 处理数据
# 将每个文本中存在的单词，按照相应位置放入 n = 10000 的向量中作为一个网络输入值
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # 创建一个（样本个数，10000） 的 0 矩阵
    for i, sequence in enumerate(sequences):
        if i == 0:
            print('第1个样本的数据值：' + str(sequence))
        results[i][sequence] = 1  # sequence是一个列表，里面是字符索引数值，将数值对应位置的0改为1

    return results


# 默认x表示样本，y表示标签
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
print('x_train:{}'.format(x_train))

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
print('y_test:{}'.format(y_test.shape))

# 从训练数据集中割取测试数据集
x_val = x_train[:10000]  # 前10000条数据作为测试数据集
partial_x_val = x_train[10000:]  # 10000以后的数据作为训练集

y_val = y_train[:10000]
partial_y_val = y_train[10000:]

# 建立模型
from keras import models, layers, optimizers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())  # 展示网络的参数
print(type(model))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=optimizers.RMSprop(lr=0.001), # 自己设定优化器参数，lr学习率
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

history = model.fit(partial_x_val, partial_y_val, batch_size=512, epochs=20, validation_data=(x_val, y_val))

# 获取训练中的具体参数值，包括 'val_loss', 'val_binary_accuracy', 'loss', 'binary_accuracy'
history_dict = history.history
print(history_dict.keys())
print(history_dict)

# 绘制图形
import matplotlib.pyplot as plt
import seaborn as sns

acc = history_dict['accuracy']  # 训练数据的准确度
val_acc = history_dict['val_accuracy']  # 验证数据的准确度
loss = history_dict['loss']  # 训练数据的损失函数值
val_loss = history_dict['val_loss']  # 验证数据的损失函数值

epochs = range(1, len(acc) + 1)  # 训练代数，从1开始

sns.set(font='FangSong', style='whitegrid')  # 简单地加上一个sns.set() 就可以设置成seaborn格式了
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure()
ax1: Axes = fig.add_subplot(2, 1, 1)
ax1.plot(epochs, loss, 'bo', label='Training loss')  # 训练损失
ax1.plot(epochs, val_loss, 'b', label='Validation loss')  # 验证损失
ax1.set_title('Training and validation loss')
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')
ax1.legend()

ax2: Axes = fig.add_subplot(2, 1, 2)
ax2.plot(epochs, acc, 'bo', label='Training acc')  # 训练精度
ax2.plot(epochs, val_acc, 'b', label='Validation acc')  # 验证精度
ax2.set_title('Training and validation accuracy')
ax2.set_xlabel('epochs')
ax2.set_ylabel('acc')
ax2.legend()
plt.show()

# 保存训练好的model
model.save('区分评论正负.h5')
