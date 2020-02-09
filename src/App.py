import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

import ColumnNames as f # feature names
from LogisticRegression import LogisticRegression

# type = 0 >> will remove all rows that have '?' value for BARE_NUCLEI feature
# type = 1 >> will replace '?' value for BARE_NUCLEI feature with 1
def pre_process(df, type=0):
    if(type == 0):
        return df[df[f.BARE_NUCLEI] != '?']
    elif(type == 1):
        return df.replace(to_replace="?", value=1)


data_set = pd.read_csv("../resources/breast-cancer-wisconsin.data.csv")
data_set.columns = [col.strip() for col in data_set.columns]

prepared_data_set = pre_process(data_set, 0)

# print(pre_processed_data_set.loc[23:28, [f.CODE, f.BARE_NUCLEI]])

# training_set = prepared_data_set[prepared_data_set[f.CLASS] == 2]
# test_set = prepared_data_set[prepared_data_set[f.CLASS] == 4]

# print(prepared_data_set.shape)
# print(training_set.shape)
# print(test_set.shape)

TARGET_FEATURE = f.BARE_NUCLEI



# _X_train = training_set.loc[:, [TARGET_FEATURE]]
# X_train = np.asarray(_X_train).astype(float)
# _pre_y_train = training_set.loc[:, [f.CLASS]]
# _y_train = np.asarray(_pre_y_train).astype(float)
# y_train = _y_train.reshape((len(_y_train),))
#
#
# _X_test = test_set.loc[:, [TARGET_FEATURE]]
# X_test = np.asarray(_X_test).astype(float)
# _pre__y_test = test_set.loc[:, [f.CLASS]]
# _y_test = np.asarray(_pre__y_test).astype(float)
# y_test = _y_test.reshape((len(_y_test),))


_X = prepared_data_set.loc[:, [TARGET_FEATURE]]
X = np.asarray(_X).astype(float)
_pre_y = prepared_data_set.loc[:, [f.CLASS]]
_y = np.asarray(_pre_y).astype(float)
y = _y.reshape((len(_y),))
#
#
# _X_test = test_set.loc[:, [TARGET_FEATURE]]
# X_test = np.asarray(_X_test).astype(float)
# _pre__y_test = test_set.loc[:, [f.CLASS]]
# _y_test = np.asarray(_pre__y_test).astype(float)
# y_test = _y_test.reshape((len(_y_test),))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, test_size=0.4)


print(X_train[:3])
print(y_train[:3])
print(X_test[:3])
print(y_test[:3])



def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


rm = LogisticRegression(learning_rate=0.01, iterations=1000, threshold=.5)
rm.fit(X_train, y_train)
predictions = rm.predict(X_test)

print("Accuracy: ", accuracy(y_test, predictions))
