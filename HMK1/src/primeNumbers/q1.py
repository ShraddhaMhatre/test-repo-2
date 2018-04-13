
# coding: utf-8

# In[690]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold

def entropy(target):
    result = 0
    types, counts = np.unique(target, return_counts=True)
    freqs = counts.astype('float')/len(target)
    for p in freqs:
        if p != 0.0:
            result -= p * np.log2(p)
    return result


# In[691]:


# read data from pdf
dataset = pd.read_csv("iris.csv")
dataset = dataset.sample(frac=1).reset_index(drop=True)

# break it into features and class
features_df = np.array(dataset.iloc[:, :-1])
class_df = np.array(dataset.iloc[:, -1])

# normalize the features
scaler = preprocessing.MinMaxScaler()
normalized_features_df = scaler.fit_transform(features_df)

#10-fold
kf = KFold(n_splits=10)

n_mins = [0.05, 0.10, 0.15, 0.20]

for n_min in n_mins:
    accuracy_arr = []
    print(n_min)
    for train, test in kf.split(dataset):
        feature_train = normalized_features_df[train]
        class_train = class_df[train]
        initial_entropy = entropy(class_train)

        train_y = []
        for e in class_train:
            train_y.append([e])

        init_train_data = np.append(feature_train, train_y, axis=1)

        max_Ig = 0
        tree = {}
        initial_training_size = np.size(feature_train, 0)

        buildTree(feature_train, class_train, initial_entropy, init_train_data, 0)

        outputs = np.array(predictor(feature_test))
        accuracy_arr.append(np.sum(outputs == class_test) *100 / np.size(class_test, 0))
    std_dav = np.std(accuracy_arr)
    avg_accuracy = np.mean(accuracy_arr)
    print(avg_accuracy)
    print(std_dav)


# In[692]:


def calcThreshold(feature, target):
    candidates = []
    sorted_feature = [feature for feature,target in sorted(zip(feature,target))]
    sorted_target = [target for feature,target in sorted(zip(feature,target))]
    sorted_f = np.array(sorted_feature)
    sorted_t = np.array(sorted_target)
    change = np.where(sorted_t[:-1] != sorted_t[1:])[0]
    for i in change:
        midpoint = float(sorted_f[i-1]) + (float(sorted_f[i]) - float(sorted_f[i-1])) / 2
        candidates.append(midpoint)
    return candidates


# In[693]:


def splitDataset(feature_index, midpoint, train_data):
    return split(train_data, train_data[:,feature_index].astype(float) < midpoint)


# In[694]:


def split(arr, cond):
    return [arr[cond], arr[~cond]]


# In[695]:


def calcNode(feature, target, midpoint, feature_index, Hq, train_data):
    splitData = splitDataset(feature_index, midpoint, train_data)
    left_dataset = splitData[0]
    right_dataset = splitData[1]
    l_classes, l_counts = np.unique(left_dataset[:,-1], return_counts=True)
    r_classes, r_counts = np.unique(right_dataset[:,-1], return_counts=True)
    Hl = entropy(left_dataset[:,-1])
    Hr = entropy(right_dataset[:,-1])
    Ig = calcIg(l_counts, r_counts, Hl, Hr, Hq)
    return(left_dataset, right_dataset, l_counts, r_counts, Hl, Hr, Ig, feature_index, l_classes, r_classes, midpoint)


# In[696]:


def calcIg(l_counts, r_counts, Hl, Hr, Hq):
    sum_l_count = np.sum(l_counts)
    sum_r_count = np.sum(r_counts)
    total = sum_l_count + sum_r_count
    return (Hq - ((sum_l_count / total) * Hl + (sum_r_count / total) * Hr))

# calcIg([3,4], [3,2], 0.9852, 0.9710, 1)
# calcIg([5,4], [1,2], 0.9911, 0.9183, 1)


# In[697]:




def calcBestNode(feature, c_train, Hq, train_data):
    midpoints = calcThreshold(feature, c_train)
    return calcNode(feature, c_train, midpoints[0], 0, Hq, train_data)

def buildTree(f_train, c_train, Hq, train_data, index):
    num_of_instances = np.size(f_train, 0)
    if(np.size(f_train,0) == 0 or np.size(c_train) == 0 or Hq == 0 or num_of_instances < n_min * initial_training_size):
        tree[index] = 0
    else:
        max_Ig = 0
        bestNode = calcBestNode(f_train[:,0], c_train, Hq, train_data)

        for f in range(np.size(f_train, 1)):
            midpoints = calcThreshold(f_train[:,f], c_train)
            for m in midpoints:
                node = calcNode(f_train[:, f], c_train, m, f, Hq, train_data)
                if(node[6] > max_Ig):
                    max_Ig = node[6]
                    bestNode = node

        tree[index] = bestNode

        buildTree(bestNode[0][:,0:(np.size(bestNode[0], 1) -1)], bestNode[0][:,-1], bestNode[4], bestNode[0], 2*index+1)
        buildTree(bestNode[1][:,0:(np.size(bestNode[1], 1) -1)], bestNode[1][:,-1], bestNode[5], bestNode[1], 2*index+2)


# In[698]:


def getMaxOutput(classes, counts):
    return classes[np.argmax(counts)]


# In[699]:


def traverse(test_tuple, index, outputs):
    feature_index = tree[index][7]
    midpoint = tree[index][10]
    l_output = tree[index][8][0]
    r_output = tree[index][9][0]
    Hl = tree[index][4]
    Hr = tree[index][5]
    l_child = tree[2*index+1]
    r_child = tree[2*index+2]
    l_classes = tree[index][8]
    r_classes = tree[index][9]
    l_counts = tree[index][2]
    r_counts = tree[index][3]
    if(test_tuple[feature_index] < midpoint):
        if(Hl == 0.0):
            outputs.append(l_output)
        elif(l_child == 0):
            outputs.append(getMaxOutput(l_classes, l_counts))
        else:
            traverse(test_tuple, 2*index+1, outputs)
    else:
        if(Hr == 0.0):
            outputs.append(r_output)
        elif(r_child == 0):
            outputs.append(getMaxOutput(r_classes, r_counts))
        else:
            traverse(test_tuple, 2*index+2, outputs)


# In[700]:


def predictor(feature_test):
    outputs = []
    if(tree == {} or tree[0] == 0):
        return None
    else:
        for e in feature_test:
            traverse(e, 0, outputs)
    
    return outputs


# In[701]:


def information_gain(original_set, attribute):
    res = entropy(original_set)
    mid_points = []

    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/len(x)

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])

    return res


# In[702]:


def is_pure(s):
    return len(set(s)) == 1


# In[703]:


def recursive_split(x, y):
    # If there could be no split, just return the original set
    if is_pure(y) or len(y) == 0:
        return y


# In[704]:


def findMidpoints(feature):
    midpoints = feature[:-1] + np.diff(feature)/2
    return midpoints


# In[705]:


x = []
x.append(3)
x.append('hello')
print(x)

