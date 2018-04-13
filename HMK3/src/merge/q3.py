import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from scipy import stats
from numpy.linalg import inv
import matplotlib.pyplot as plt

# read data from pdf
dataset1 = pd.read_csv("housing.csv", header=None)
dataset2 = pd.read_csv("yachtData.csv", header=None)


def calc_SSE(w, x, y):
    return np.sum(np.square(np.dot(x, w) - y[np.newaxis].T))


def calc_RMSE(w, x, y):
    return np.sqrt(calc_SSE(w, x, y) / np.size(x, 0))


def zscore(feature, f_mean, f_std):
    return (feature - f_mean) / f_std

def grad_descent(w, x, y, max_itr, RMSE, tolerance, learning_rate):
    for i in range(0, max_itr):
        w = w - learning_rate * np.sum(np.multiply(np.dot(x, w) - y[np.newaxis].T, x), axis=0)[np.newaxis].T
        new_RMSE = calc_RMSE(w, x, y)
        if abs(RMSE - new_RMSE) <= tolerance:
            break
        else:
            RMSE = new_RMSE

    return w

def gradient_descent(x, y):
    return np.dot(np.dot(inv(np.dot(x.T, x)), x.T), y[np.newaxis].T)


def run(dataset, lr, tol):
    # shuffle the dataset
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    # convert dataframe into numpy array
    np_data = dataset.values

    train_RMSE_arr = []
    test_RMSE_arr = []

    # 10-fold
    kf = KFold(n_splits=10)
    for train, test in kf.split(np_data):
        train_data = np_data[train]
        test_data = np_data[test]

        feature_train = train_data[:, :-1]
        class_train = train_data[:, -1]

        feature_stats = stats.describe(feature_train)

        feature_test = test_data[:, :-1]
        class_test = test_data[:, -1]

        # z score
        for f in range(0, np.size(feature_train, 1)):
            feature_train[:, f] = zscore(feature_train[:, f], feature_stats.mean[f], np.sqrt(feature_stats.variance[f]))
            feature_test[:, f] = zscore(feature_test[:, f], feature_stats.mean[f], np.sqrt(feature_stats.variance[f]))

        add_feature_train = np.ones((np.size(feature_train, 0), 1))
        add_feature_test = np.ones((np.size(feature_test, 0), 1))
        feature_train = np.append(add_feature_train, feature_train, axis=1)
        feature_test = np.append(add_feature_test, feature_test, axis=1)

        learning_rate = lr
        tolerance = tol
        max_itr = 1000
        init_w = np.zeros((np.size(feature_train, 1), 1))
        init_RMSE = calc_RMSE(init_w, feature_train, class_train)

        weight_vector1 = gradient_descent(feature_train, class_train)
        weight_vector2 = grad_descent(init_w, feature_train, class_train, max_itr, init_RMSE, tolerance, learning_rate)

        train_RMSE = calc_RMSE(weight_vector1, feature_train, class_train)
        train_RMSE_arr.append(train_RMSE)
        test_RMSE = calc_RMSE(weight_vector2, feature_test, class_test)
        test_RMSE_arr.append(test_RMSE)

    plt.scatter(np.arange(1, 11), train_RMSE_arr, c="r")
    plt.scatter(np.arange(1, 11), test_RMSE_arr, c="b")
    plt.ylabel("RMSE")
    plt.xlabel("Folds")
    plt.show()


run(dataset1, 0.0004, 0.005)
run(dataset2, 0.001, 0.001)
