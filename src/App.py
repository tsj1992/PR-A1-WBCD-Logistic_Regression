import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt

import ColumnNames as f  # feature names
from LogisticRegression import LogisticRegression


# type = 0 >> will remove all rows that have '?' value for BARE_NUCLEI feature
# type = 1 >> will replace '?' value for BARE_NUCLEI feature with 1
def pre_process(df, type=0):
    if (type == 0):
        return df[df[f.BARE_NUCLEI] != '?']  # remove rows with missing data
    elif (type == 1):
        return df.replace(to_replace="?", value=1)  # replace 1 for missing data


# to calculate the accuracy of the model
def accuracy(y_test, y_pred):
    y_true = [1 if i == 4 else 0 for i in y_test]
    return np.sum(y_true == y_pred) / len(y_true)


data_set = pd.read_csv("../resources/breast-cancer-wisconsin.data.csv")  # get data
data_set.columns = [col.strip() for col in data_set.columns]  # strip whitespaces of column names

prepared_data_set = pre_process(data_set, 1)  # pre process data

# x = prepared_data_set.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# df = pd.DataFrame(x_scaled)


COLUMNS = [
    f.CLUMP_THICKNESS,
    f.UNIFORMITY_OF_CELL_SIZE,
    f.UNIFORMITY_OF_CELL_SHAPE,
    f.MARGINAL_ADHESION,
    f.SINGLE_EPITHELIAL_CELL_SIZE,
    f.BARE_NUCLEI,
    f.BLAND_CHROMATIN,
    f.NORMAL_NUCLEOLI,
    f.MITOSES
]

print("|---------------------------------------------------------|")
print("| {:<5} {:<25}|{:<3} {:<19} |".format(" ", "FEATURE", " ", "ACCURACY"))
print("|---------------------------------------------------------|")

for column in COLUMNS:
    _X = prepared_data_set.loc[:, [column]]
    X = np.asarray(_X).astype(float)
    _pre_y = prepared_data_set.loc[:, [f.CLASS]]
    _y = np.asarray(_pre_y).astype(float)
    y = _y.reshape((len(_y),))

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.65, test_size=0.35)

    rm = LogisticRegression(learning_rate=0.001, iterations=10000, threshold=.5)
    rm.fit(X_train, y_train)
    predictions = rm.predict(X_test)

    print("| {:<30} | {:<20} % |".format(column, accuracy(y_test, predictions) * 100))

print("|---------------------------------------------------------|")
