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


def kflod():
    # 使用K折交叉验证来 训练模型
    import numpy as np

    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 500
    all_mae_histories = []
    all_history = []
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
        history = model.fit(
            partial_train_data, partial_train_targets,
            validation_data=(val_data, val_targets),
            epochs=num_epochs, batch_size=1, verbose=2)
        # verbose = 0 为不在标准输出流输出日志信息
        # verbose = 1 为输出进度条记录
        # verbose = 2 为每个epoch输出一行记录

        # 获取验证集上的 评价指标
        all_history.append(history.history)
        print(history.history.keys())
        all_mae_histories.append(history.history['val_mae'])

    # 计算所有轮次中的k折验证分数平均值
    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
    print('average_mae_history的长度：', len(average_mae_history))
    print(average_mae_history)
    print('输出第几轮得到最小的mae：', np.argmin(average_mae_history))

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style='white', font='FangSong')

    fig1 = plt.figure()
    plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()


# 见09-指数平滑，可知在第64轮，mae最小，所以最后训练时驯良64轮
# 在实际训练时发现，90轮比64轮更好

model = build_model()
model.fit(train_data, train_targets, epochs=90, batch_size=16, verbose=1)

model.save('房价回归最终模型.h5')

test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print('test_mse_score:' + str(test_mse_score))
print('test_mae_score:' + str(test_mae_score))
