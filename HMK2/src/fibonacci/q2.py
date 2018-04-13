import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from scipy import stats
import matplotlib.pyplot as plt

# read data from pdf
dataset1 = pd.read_csv("housing.csv", header=None)
dataset2 = pd.read_csv("yachtData.csv", header=None)
dataset3 = pd.read_csv("concreteData.csv", header=None)


def calc_SSE(w, x, y):
    return np.sum(np.square(np.dot(x, w) - y[np.newaxis].T))


def calc_RMSE(w, x, y):
    return np.sqrt(calc_SSE(w, x, y) / np.size(x, 0))


def zscore(feature, f_mean, f_std):
    return (feature - f_mean) / f_std


def gradient_descent(w, x, y, max_itr, RMSE, tolerance, learning_rate):
    rmse_list = []
    for i in range(0, max_itr):
        w = w - learning_rate * np.sum(np.multiply(np.dot(x, w) - y[np.newaxis].T, x), axis=0)[np.newaxis].T
        new_RMSE = calc_RMSE(w, x, y)
        if abs(RMSE - new_RMSE) <= tolerance:
            break
        else:
            RMSE = new_RMSE
            rmse_list.append(RMSE)
    return w, rmse_list


def run(dataset, lr, tol, dataset_name):
    print(dataset_name + ' Dataset')

    # shuffle the dataset
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    # convert dataframe into numpy array
    np_data = dataset.values

    train_SSE_arr = []
    test_SSE_arr = []



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

        for f in range(0, np.size(feature_train, 1)):
            feature_train[:, f] = zscore(feature_train[:, f], feature_stats.mean[f], np.sqrt(feature_stats.variance[f]))
            feature_test[:, f] = zscore(feature_test[:, f], feature_stats.mean[f], np.sqrt(feature_stats.variance[f]))

        add_feature_train = np.ones((np.size(feature_train, 0), 1))
        add_feature_test = np.ones((np.size(feature_test, 0), 1))
        feature_train = np.append(add_feature_train, feature_train, axis=1)
        feature_test = np.append(add_feature_test, feature_test, axis=1)

        init_w = np.zeros((np.size(feature_train, 1), 1))

        learning_rate = lr
        tolerance = tol
        max_itr = 1000

        init_RMSE = calc_RMSE(init_w, feature_train, class_train)

        weight_vector, rmse_list = gradient_descent(init_w, feature_train, class_train, max_itr, init_RMSE, tolerance, learning_rate)
        RMSE_per_fold.append(rmse_list)

        train_SSE = calc_SSE(weight_vector, feature_train, class_train)
        train_SSE_arr.append(train_SSE)
        train_RMSE = calc_RMSE(weight_vector, feature_train, class_train)
        train_RMSE_arr.append(train_RMSE)

        test_SSE = calc_SSE(weight_vector, feature_test, class_test)
        test_SSE_arr.append(test_SSE)
        test_RMSE = calc_RMSE(weight_vector, feature_test, class_test)
        test_RMSE_arr.append(test_RMSE)

    avg_train_SSE = np.average(train_SSE_arr)
    avg_train_RMSE = np.average(train_RMSE_arr)

    avg_test_SSE = np.average(test_SSE_arr)
    avg_test_RMSE = np.average(test_RMSE_arr)

    std_train_SSE = np.std(train_SSE_arr)
    std_train_RMSE = np.std(train_RMSE_arr)

    std_test_SSE = np.std(test_SSE_arr)
    std_test_RMSE = np.std(test_RMSE_arr)

    print()
    print('Train Data:')
    print('SSE across 10 folds:')
    print(train_SSE_arr)
    print('RMSE across10 folds:')
    print(train_RMSE_arr)
    print('Average SSE:')
    print(avg_train_SSE)
    print('Average RMSE:')
    print(avg_train_RMSE)
    print('SSE Standard Deviation:')
    print(std_train_SSE)
    print('RMSE Standard Deviation:')
    print(std_train_RMSE)
    print()

    print('Test Data:')
    print('SSE across 10 folds:')
    print(test_SSE_arr)
    print('RMSE across 10 folds:')
    print(test_RMSE_arr)
    print('avg_test_SSE:')
    print(avg_test_SSE)
    print('avg_test_RMSE:')
    print(avg_test_RMSE)
    print('SSE Standard Deviation:')
    print(std_test_SSE)
    print('RMSE Standard Deviation:')
    print(std_test_RMSE)


train_RMSE_arr = []
test_RMSE_arr = []
RMSE_per_fold = []
run(dataset1, 0.0004, 0.005, 'Housing')
print(len(RMSE_per_fold))
plt.plot(np.arange(0, len(RMSE_per_fold[2])), RMSE_per_fold[2], c="r")
# plt.plot(np.arange(1, 11), test_RMSE_arr, c="b")
plt.ylabel("RMSE")
plt.xlabel("Iterations")
plt.show()

train_RMSE_arr = []
test_RMSE_arr = []
RMSE_per_fold = []
run(dataset2, 0.001, 0.001, 'Yacht')
plt.plot(np.arange(0, len(RMSE_per_fold[2])), RMSE_per_fold[2], c="r")
plt.ylabel("RMSE")
plt.xlabel("folds")
plt.show()

train_RMSE_arr = []
test_RMSE_arr = []
RMSE_per_fold = []
run(dataset3, 0.0007, 0.0001, 'Concrete')
plt.plot(np.arange(0, len(RMSE_per_fold[2])), RMSE_per_fold[2], c="r")
plt.ylabel("RMSE")
plt.xlabel("Folds")
plt.show()

