import pandas as pd
import numpy as np
from scipy import stats
from numpy.linalg import inv
import matplotlib.pyplot as plt

# read data from pdf
dataset1 = pd.read_csv("sinData_Train.csv", header=None)
validation_dataset = pd.read_csv("sinData_Validation.csv", header=None)
# dataset2 = pd.read_csv("yatchData.csv", header=None)
# dataset3 = pd.read_csv("concreteData.csv", header=None)


def calc_SSE(w, x, y):
    return np.sum(np.square(np.dot(x, w) - y[np.newaxis].T))


def calc_RMSE(w, x, y):
    return np.sqrt(calc_SSE(w, x, y) / np.size(x, 0))


def gradient_descent(x, y):
    return np.dot(np.dot(inv(np.dot(x.T, x)), x.T), y[np.newaxis].T)


def run(dataset, validation_dataset, p):
    # shuffle the dataset
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    validation_dataset = validation_dataset.sample(frac=1).reset_index(drop=True)

    # convert dataframe into numpy array
    np_data = dataset.values
    validation_np_data = validation_dataset.values

    feature_train = np_data[:, :-1]
    class_train = np_data[:, -1]

    validation_feature = validation_np_data[:, :-1]
    validation_class = validation_np_data[:, -1]

    expanded_feature_train = feature_train
    expanded_validation_feature = validation_feature
    # print(expanded_feature_train)

    # a = np.array([[1], [2], [3]])
    # print(a)
    # b = np.array([4, 5, 6])
    # print(b)
    # print(np.column_stack((a, b)))

    if p > 1:
        for power in range(2, p + 1):
            for f in range(np.size(feature_train, 1)):
                expanded_feature_train = np.column_stack((expanded_feature_train, np.power(feature_train[:, f], power)))
                expanded_validation_feature = np.column_stack((expanded_validation_feature, np.power(validation_feature[:, f], power)))


    # print(expanded_feature_train)
    # print(np.size(np.square(feature_train[:, 0])))
    # print(np.column_stack((expanded_feature_train, np.square(feature_train[:, 0]))))

    add_feature_train = np.ones((np.size(expanded_feature_train, 0), 1))
    expanded_feature_train = np.append(add_feature_train, expanded_feature_train, axis=1)

    validation_add_feature = np.ones((np.size(expanded_validation_feature, 0), 1))
    expanded_validation_feature = np.append(validation_add_feature, expanded_validation_feature, axis=1)

    # print(expanded_feature_train)
    weight_vector = gradient_descent(expanded_feature_train, class_train)
    # print(weight_vector)
    # print('Train_sse: ')
    train_SSE = calc_SSE(weight_vector, expanded_feature_train, class_train)
    train_SSE_arr.append(train_SSE)
    train_RMSE = calc_RMSE(weight_vector, expanded_feature_train, class_train)
    # print(train_RMSE)
    # print('Validation_sse: ')
    validation_SSE = calc_SSE(weight_vector, expanded_validation_feature, validation_class)
    validation_SSE_arr.append(validation_SSE)
    validation_RMSE = calc_RMSE(weight_vector, expanded_validation_feature, validation_class)
    # print(validation_RMSE)
    # print(validation_RMSE)
    # print(np.dot(expanded_feature_train, weight_vector))
    # print(np.sqrt(np.sum(np.square(np.dot(feature_train, weight_vector) - class_train[np.newaxis].T)) / np.size(class_train, 0)))
    # print(test_RMSE)
    # print('pow: ')
    # print(pow)
    # print()


max_p = 15
train_SSE_arr = []
validation_SSE_arr = []

for po in range(1, max_p + 1):
    run(dataset1, validation_dataset, po)
# run(dataset2)
# run(dataset3)
print(train_SSE_arr)
print(validation_SSE_arr)

plt.scatter(np.arange(1, max_p + 1), train_SSE_arr, c="r")
plt.scatter(np.arange(1, max_p + 1), validation_SSE_arr, c="b")
plt.ylabel("Average SSE")
plt.xlabel("powers")
plt.show()
