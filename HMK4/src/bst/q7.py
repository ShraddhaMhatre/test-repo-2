import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from numpy.linalg import inv
import matplotlib.pyplot as plt

# read data from pdf
dataset = pd.read_csv("sinData_Train.csv", header=None)


def calc_SSE(w, x, y):
    return np.sum(np.square(np.mean(y) + np.dot(x, w) - y[np.newaxis].T))


def calc_RMSE(w, x, y):
    return np.sqrt(calc_SSE(w, x, y) / np.size(x, 0))


def zscore(feature, f_mean, f_std):
    return (feature - f_mean) / f_std


def gradient_descent(x, y, lamb):
    return np.dot(np.dot(inv(np.dot(x.T, x) + lamb * np.identity(np.size(x, 1))), x.T), y[np.newaxis].T)


def center_data(np_data):
    mean_arr = np.mean(np_data, axis=0)
    np_data = np.subtract(np_data, mean_arr)
    return np_data


def run(dataset, p, lamb):

    # shuffle the dataset
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    # convert dataframe into numpy array

    # print(center_data([[1, 2], [3, 4]]))
    np_data = dataset.values
    # print(np_data)
    # print()
    # center all instances of features and classes
    np_data = center_data(np_data)
    # print(np_data)

    train_RMSE_arr = []
    test_RMSE_arr = []
    # 10-fold
    kf = KFold(n_splits=10)
    for train, test in kf.split(np_data):
        train_data = np_data[train]
        test_data = np_data[test]

        feature_train = train_data[:, :-1]
        class_train = train_data[:, -1]

        feature_test = test_data[:, :-1]
        class_test = test_data[:, -1]

        expanded_feature_train = feature_train
        expanded_feature_test = feature_test

        if p > 1:
            for power in range(2, p + 1):
                for f in range(np.size(feature_train, 1)):
                    expanded_feature_train = np.column_stack((expanded_feature_train, np.power(feature_train[:, f], power)))
                    expanded_feature_test = np.column_stack((expanded_feature_test, np.power(feature_test[:, f], power)))

        weight_vector = gradient_descent(expanded_feature_train, class_train, lamb)

        train_RMSE = calc_RMSE(weight_vector, expanded_feature_train, class_train)
        train_RMSE_arr.append(train_RMSE)

        test_RMSE = calc_RMSE(weight_vector, expanded_feature_test, class_test)
        test_RMSE_arr.append(test_RMSE)

    avg_train_RMSE_arr.append(np.average(train_RMSE_arr))
    avg_test_RMSE_arr.append(np.average(test_RMSE_arr))


lambs = np.arange(0, 10.2, 0.2)

avg_train_RMSE_arr = []
avg_test_RMSE_arr = []
for lamb in lambs:
    run(dataset, 5, lamb)

plt.scatter(lambs, avg_train_RMSE_arr, c="r")
plt.scatter(lambs, avg_test_RMSE_arr, c="b")
plt.ylabel("Average RMSE")
plt.xlabel("powers")
plt.show()

avg_train_RMSE_arr = []
avg_test_RMSE_arr = []
for lamb in lambs:
    run(dataset, 9, lamb)

plt.scatter(lambs, avg_train_RMSE_arr, c="r")
plt.scatter(lambs, avg_test_RMSE_arr, c="b")
plt.ylabel("Average RMSE")
plt.xlabel("powers")
plt.show()
