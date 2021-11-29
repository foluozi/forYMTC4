from keras.datasets import mnist
import numpy as np

# 下载mnist手写字集合，以
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images: np.ndarray
train_labels: np.ndarray
test_images: np.ndarray
test_labels: np.ndarray

print(train_images.shape)
print(type(train_images))
print(train_images[0])
print(len(train_labels))
print(train_labels)

# # 使用Matplotlib显示图片
# import matplotlib.pyplot as plt
#
# plt.imshow(train_images[0], cmap=plt.cm.binary)
# plt.show()

from keras import models
from keras import layers

network = models.Sequential(name='first ANN')  # 初始化一份模型

# 构建模型，这是一个两层的模型
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))  # 这是第一个隐藏层（也是这个模型中唯一一个），后面的imput_shape相当于是输入层
network.add(layers.Dense(10, activation='softmax'))  # 这是输出层
print(network.summary())

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape((len(train_images), 28 * 28))  # 将三维张量拉成二维张量
train_images = train_images.astype('float32') / 255  # 做归一化

test_images = test_images.reshape((len(test_images), 28 * 28))  # 将三维张量拉成二维张量
test_images = test_images.astype('float32') / 255  # 做归一化

from keras.utils import to_categorical

# 把标签值转换成一维数组，例如 2 --> [0,0,1,0,0,0,0,0,0,0]
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

network.fit(train_images, train_labels, epochs=6, batch_size=128)

# 获取test数据的损失值和正确度
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss:{}'.format(test_loss))
print('test_acc:{}'.format(test_acc))
