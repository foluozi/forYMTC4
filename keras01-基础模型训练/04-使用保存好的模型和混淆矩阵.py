import keras
from keras.datasets import imdb
from keras.models import load_model
import numpy as np
from keras import models

# 导入模型
model: models = load_model('区分评论正负.h5')

print(type(model))
# 获取数据
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # 创建一个（样本个数，10000） 的 0 矩阵
    for i, sequence in enumerate(sequences):
        if i == 0:
            print('第1个样本的数据值：' + str(sequence))
        results[i][sequence] = 1  # sequence是一个列表，里面是字符索引数值，将数值对应位置的0改为1

    return results


x_test = vectorize_sequences(test_data)
y_test = np.asarray(test_labels).astype('float32')

pred_10 = model.predict(x_test[:10])  # 给出sigmoid值
pred_class_10 = model.predict_classes(x_test[:10])  # 分类

print(pred_10)
print(pred_class_10)

pred_class_all = model.predict_classes(x_test)
print(pred_class_all.shape)  # (25000,1)  二维张量
print(y_test.shape)  # (25000,)  一维张量

predictionre_T = pred_class_all.reshape(-1)  # reshape(-1) 相当于转置  将pred_class_all变成了(25000，）
print(y_test, predictionre_T)

import pandas as pd

# 创建混淆矩阵(Confusion Matrix)  详细解析可见  https://zhuanlan.zhihu.com/p/46204175
print(pd.crosstab(y_test, predictionre_T, rownames=['label'], colnames=['predict']))
# predict     0      1
# label
# 0.0      8999   3501
# 1.0       772  11728

from sklearn.metrics import classification_report
# 查看模型预测的  精度（precision）  召回率（recall）  F1值（F1-score）
print(classification_report(y_test, pred_class_all))


#               precision    recall  f1-score   support
#
#          0.0       0.92      0.72      0.81     12500
#          1.0       0.77      0.94      0.85     12500
#
#     accuracy                           0.83     25000
#    macro avg       0.85      0.83      0.83     25000
# weighted avg       0.85      0.83      0.83     25000
