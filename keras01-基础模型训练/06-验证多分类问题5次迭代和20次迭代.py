from keras.models import load_model
import keras
from keras.datasets import reuters
import numpy as np
from keras.utils.np_utils import to_categorical

model20 = load_model('多分类问题.h5')
model5 = load_model('多分类问题_epochs=5.h5')

# 数据处理
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


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

pred20 = model20.predict_classes(x_test)
pred5 = model5.predict_classes(x_test)

print(test_labels[[0, 1, 2, 3, 4]])
print(pred5[[0, 1, 2, 3, 4]])

print(test_labels.shape)
print(pred20.shape)
print(test_labels.max(),test_labels.min())
print(pred20.max(), pred20.min())
print(pred5.max(),pred5.min())

import pandas as pd

print(pd.crosstab(test_labels, pred5, rownames=['label'], colnames=['predict']))
print(pd.crosstab(test_labels, pred20, rownames=['label'], colnames=['predict']))
print(pd.crosstab(pred5,pred20,rownames=['5'], colnames=['20']))

from sklearn.metrics import classification_report

print(classification_report(test_labels, pred5))
print(classification_report(test_labels, pred20))
