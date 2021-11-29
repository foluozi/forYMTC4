import keras
from keras.datasets import boston_housing

# 获取数据
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape)
print(train_targets.shape)
print(test_data)
print(test_targets)

# 处理数据
# 正态化处理，中心值为0
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)  # 标准差
train_data /= std

test_data -= mean
test_data /= std

print(mean)
print(train_data)
print(test_data)

from keras import models, layers


def build_model():
    model = models.Sequential()

    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))

    # 此次loss为mse（均方误差），预测值与目标值之差的平方。回归问题常用的损失函数
    # 监控指标为mae（平均绝对误差），预测值与目标值之差的绝对值。
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# 使用K折交叉验证来 训练模型
import numpy as np

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
all_mse = []
for i in range(k):
    print('processing fold # ', i)
    # 将第i组数据作为验证集
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # 将除了第i组之外的数据作为训练数据
    partial_train_data = np.concatenate(  # np.concatenate 在某个维度上连接两个张量
        [train_data[: i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0
    )
    partial_train_targets = np.concatenate(
        [train_targets[: i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0
    )

    model = build_model()
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=2)
    # verbose = 0 为不在标准输出流输出日志信息
    # verbose = 1 为输出进度条记录
    # verbose = 2 为每个epoch输出一行记录

    # 获取验证集上的 评价指标
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=2)

    all_scores.append(val_mae)
    all_mse.append(val_mse)

print(all_scores)
print(all_mse)

print(np.mean(all_scores))
print(np.mean(all_mse))
