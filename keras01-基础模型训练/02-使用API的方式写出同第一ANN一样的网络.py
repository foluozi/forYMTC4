from keras import layers, models

# 输入层28*28个神经元
input_tensor = layers.Input(shape=(28 * 28,))
# 隐藏层32个神经元
x = layers.Dense(32, activation='relu')(input_tensor)  # 相当于把layers.Dense(32, activation='relu')()当做一个函数
# 输出层10个神经元
output_tensor = layers.Dense(10, activation='softmax')(x)
# 构建模型，指定输入输出
model = models.Model(inputs=input_tensor, outputs=output_tensor)

from keras import optimizers

# 指定优化器为RMS，学习速率为0.01 ， lose, metrics
model.compile(optimizer=optimizers.RMSprop(lr=0.01), loss='mse', metrics=['accuracy'])


# 处理数据
from keras.datasets import mnist
import numpy as np

train_images: np.ndarray
train_labels: np.ndarray
test_images: np.ndarray
test_labels: np.ndarray
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((len(train_images), 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((len(test_images), 28 * 28))
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# 运行network
model.fit(train_images, train_labels, batch_size=128, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_loss:{}'.format(test_loss))  # test_loss:0.006775414498857542
print('test_acc:{}'.format(test_acc))  # test_acc:0.9603000283241272