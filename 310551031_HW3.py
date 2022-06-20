#!/usr/bin/env python
# coding: utf-8

# ## HW3: Decision Tree, AdaBoost and Random Forest
# In hw3, you need to implement decision tree, adaboost and random forest by using only numpy, then train your implemented model by the provided dataset and test the performance with testing data
#
# Please note that only **NUMPY** can be used to implement your model, you will get no points by simply calling sklearn.tree.DecisionTreeClassifier

# ## Load data
# The dataset is the Heart Disease Data Set from UCI Machine Learning Repository. It is a binary classifiation dataset, the label is stored in `target` column. **Please note that there exist categorical features which need to be [one-hot encoding](https://www.datacamp.com/community/tutorials/categorical-data) before fit into your model!**
# See follow links for more information
# https://archive.ics.uci.edu/ml/datasets/heart+Disease

# In[2]:


from unicodedata import category
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from collections import Counter

file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
df = pd.read_csv(file_url)

train_idx = np.load('train_idx.npy')
test_idx = np.load('test_idx.npy')

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

# train_df.loc[train_df["target"] == 0, "target"] = -1
# test_df.loc[test_df["target"] == 0, "target"] = -1

# print("training shape:", train_df.shape)
# print("testing shape:", test_df.shape)

# print(train_df)
# print(test_df)


# In[3]:

# train_df.head()
# test_df.head()

# print(list(train_df))
# print("age:", set(test_df['age']))
# print("sex:", set(test_df['sex']))
# print("cp:", set(test_df['cp']))
# print("trestbps:", set(test_df['trestbps']))
# print("chol:", set(test_df['chol']))
# print("fbs:", set(test_df['fbs']))
# print("restecg:", set(test_df['restecg']))
# print("thalach:", set(test_df['thalach']))
# print("exang:", set(test_df['exang']))
# print("oldpeak:", set(test_df['oldpeak']))
# print("slope:", set(test_df['slope']))
# print("ca:", set(test_df['ca']))
# print("thal:", set(test_df['thal']))
# print("target:", set(test_df['target']))

# categorical value

# sex: {0, 1}
# cp: {0, 1, 2, 3, 4}
# fbs: {0, 1}
# restecg: {0, 1, 2}
# exang: {0, 1}
# slope: {1, 2, 3}
# ca: {0, 1, 2, 3}
# thal: {'fixed', 'reversible', 'normal'}

# numerical value

# age: {29, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
#       53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 74, 77}
# trestbps: {128, 130, 132, 134, 135, 136, 138, 140, 142, 144, 145, 146, 148, 150, 152, 154, 155, 158, 160, 165, 170,
#            174, 178, 180, 192, 200, 100, 101, 102, 104, 105, 106, 108, 110, 112, 115, 117, 118, 120, 122, 123, 124, 125, 126}
# chol: {126, 131, 141, 160, 164, 166, 167, 168, 172, 175, 177, 180, 182, 183, 185, 186, 188, 193, 197, 198, 201, 203, 204, 205, 206, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 219, 220, 221, 222, 223, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 239, 240, 241, 242, 243,
#        244, 245, 246, 248, 249, 250, 254, 255, 256, 258, 259, 260, 261, 263, 264, 265, 266, 267, 268, 269, 271, 273, 274, 275, 278, 281, 282, 283, 284, 286, 288, 289, 290, 294, 295, 298, 299, 302, 303, 304, 305, 306, 308, 309, 311, 313, 315, 318, 319, 326, 327, 330, 335, 340, 342, 353, 360, 394, 407}
# thalach: {128, 130, 131, 132, 133, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
#           172, 173, 174, 175, 178, 179, 180, 181, 182, 184, 185, 186, 187, 188, 190, 192, 195, 71, 202, 95, 96, 97, 103, 105, 106, 108, 109, 111, 113, 114, 115, 116, 117, 118, 120, 121, 122, 123, 124, 125, 126, 127}
# oldpeak: {0.0, 0.8, 2.8, 3.6, 1.4, 1.2, 0.2, 1.9, 1.0, 3.2, 4.0, 3.0, 6.2, 2.0, 0.5, 2.5, 3.5,
#           0.6, 5.6, 1.5, 1.1, 1.6, 2.6, 0.7, 0.1, 2.2, 2.3, 0.3, 3.8, 4.2, 1.8, 1.3, 0.4, 0.9, 2.4, 3.4}

# target
# target: {0, 1}

# ## Question 1
# Gini Index or Entropy is often used for measuring the “best” splitting of the data. Please compute the Entropy and Gini Index of provided data. Please use the formula from [page 5 of hw3 slides](https://docs.google.com/presentation/d/1kIe_-YZdemRMmr_3xDy-l0OS2EcLgDH7Uan14tlU5KE/edit#slide=id.gd542a5ff75_0_15)

# In[5]:


def gini(sequence):
    labels, count = np.unique(sequence, return_counts=True)
    probabilities = count/len(sequence)
    return 1-np.sum(probabilities**2)


def entropy(sequence):
    labels, count = np.unique(sequence, return_counts=True)
    probabilities = count/len(sequence)
    return -1*np.sum(probabilities*np.log2(probabilities))


def accuracy(predict, answer):
    predict = np.array(predict)
    answer = np.array(answer)
    # answer = answer.reshape(-1,)
    # print("predict:", predict)
    # print("answer:", answer)
    # predict[predict == -1] = 0
    # answer[answer == -1] = 0
    return np.sum(predict == answer)/len(answer)

# In[14]:


criterions = {
    'gini': gini,
    'entropy': entropy
}

# 1 = class 1,
# 2 = class 2
data = np.array([1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2])

print("-------------------------------------------------------------------------------")

# In[15]:
print("Gini of data is ", gini(data))


# In[16]:
print("Entropy of data is ", entropy(data))


# ## Question 2
# Implement the Decision Tree algorithm (CART, Classification and Regression Trees) and trained the model by the given arguments, and print the accuracy score on the test data. You should implement two arguments for the Decision Tree algorithm
# 1. **criterion**: The function to measure the quality of a split. Your model should support `gini` for the Gini impurity and `entropy` for the information gain.
# 2. **max_depth**: The maximum depth of the tree. If `max_depth=None`, then nodes are expanded until all leaves are pure. `max_depth=1` equals to split data once
#

# In[7]:


class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion_name = criterion
        self.criterion = criterions[criterion]
        self.max_depth = max_depth
        self.root = None
        self.x = None
        self.y = None
        self.feature_names = None
        self.importance_dict = None
        # self.used_attribute = set()

    def fit(self, x, y):
        self.feature_names = list(x.columns)
        self.importance_dict = dict(
            zip(self.feature_names, [0]*len(self.feature_names)))
        self.x = x
        self.y = y
        x_copy = x.copy()
        y_copy = y.copy()
        x_copy['target'] = y_copy
        self.root = TreeNode(x_copy, y_copy, self, 0)
        self.root.BuildTree()

        return None

    def predict(self, x):
        y_pred = []
        for index in range(len(x)):
            Node = self.root
            xdata = x.iloc[index, :]
            while ((Node.leftNode is not None) or (Node.rightNode is not None)):
                if(xdata[Node.criterion_attr] <= Node.criterion_attr_value):
                    Node = Node.leftNode
                elif(xdata[Node.criterion_attr] > Node.criterion_attr_value):
                    Node = Node.rightNode
            if Node.predict is None:
                Node = Node.parent
            y_pred.append(Node.predict)
        return y_pred

    def Show(self):
        '''
        show shape of tree
        Left {Middle} Right
        '''
        self.root.Show()

    def get_feature(self, Node):

        if(Node.criterion_attr in self.importance_dict):
            self.importance_dict[Node.criterion_attr] += 1
        if((Node.leftNode is not None) and (Node.criterion_attr in self.importance_dict)):
            self.get_feature(Node.leftNode)
        if((Node.rightNode is not None) and (Node.criterion_attr in self.importance_dict)):
            self.get_feature(Node.rightNode)

    def plot_feature_importance(self):

        self.get_feature(self.root)

        plt.barh(list(self.importance_dict.keys()),
                 list(self.importance_dict.values()))
        plt.ylabel('feature names')
        plt.xlabel('feature importance')
        plt.yticks(list(self.importance_dict.keys()),
                   list(self.importance_dict.keys()))
        plt.grid(True)
        plt.savefig('feature_importance.png')
        # plt.show()


class TreeNode():
    def __init__(self, x, y, DT: DecisionTree, depth):
        self.x = x
        self.y = y
        # self.index = index
        self.DT = DT
        self.depth = depth
        self.min_criterion_value = 999999
        self.criterion_attr_value = None
        self.criterion_attr = None
        self.predict = None
        self.parent = None
        self.leftNode = None
        self.rightNode = None
        self.criterion = self.DT.criterion(list(self.x["target"]))

    def SplitAttribute(self):

        self.min_criterion_value = 999999
        self.criterion_attr_value = None
        self.criterion_attr = None

        # print(self.DT.feature_names)
        # feature_list = []
        # feature_list_gini = []
        for attr in self.DT.feature_names:
            # total_category_value = np.unique(self.x[attr])
            total_category_value = list(set(self.x[attr]))
            total_category_value = sorted(total_category_value)
            # new_total_category_value = [
            #     (total_category_value[idx]+total_category_value[idx+1])/2 for idx in range(len(total_category_value)-1)]
            # total_category_value = new_total_category_value
            # total_category_value = [total_category_value[0]] + \
            #     new_total_category_value + [total_category_value[-1]]
            # total_category_value += new_total_category_value
            # total_category_value = sorted(total_category_value)
            # print(total_category_value)
            for value in total_category_value:

                A = self.x[self.x[attr] <= value]
                B = self.x[self.x[attr] > value]

                total = self.x[attr]

                weightA = len(A)/len(total)
                weightB = len(B)/len(total)

                LabelA = A["target"]
                LabelB = B["target"]

                # print("LabelA:", list(LabelA), "LabelB:", list(LabelB))

                criterion_value = weightA * \
                    self.DT.criterion(LabelA) + weightB * \
                    self.DT.criterion(LabelB)

                # if(criterion_value < 0):
                #     print("criterion_value less than zero.")
                # feature_list.append([attr, value])
                # feature_list_gini.append(criterion_value)
                if(criterion_value < self.min_criterion_value):
                    self.min_criterion_value = criterion_value
                    # self.criterion_categories = self.DT.categorys[attr]
                    self.criterion_attr_value = value
                    self.criterion_attr = attr

        # print("self.x:", self.x)
        # print("self.criterion_attr:", self.criterion_attr)
        # print("self.criterion_attr_value:", self.criterion_attr_value)

    def BuildTree(self):

        if len(self.x) == 1:
            final_predict = list(self.x["target"])
            self.predict = max(set(final_predict),
                               key=final_predict.count)
            return

        if (self.DT.max_depth is not None) and (self.depth == self.DT.max_depth):
            final_predict = list(self.x["target"])
            self.predict = max(set(final_predict),
                               key=final_predict.count)
            return

        if self.criterion == 0:
            final_predict = list(self.x["target"])
            self.predict = max(set(final_predict),
                               key=final_predict.count)
            return

        self.SplitAttribute()

        # if len(self.x) > 1:
        #     final_predict = list(self.x["target"])
        #     self.predict = max(set(final_predict),
        #                        key=final_predict.count)

        # final_predict = list(self.x["target"])
        # self.predict = max(set(final_predict),
        #                    key=final_predict.count)
        A = self.x[self.x[self.criterion_attr] <= self.criterion_attr_value]
        B = self.x[self.x[self.criterion_attr] > self.criterion_attr_value]

        LabelA = A['target']
        LabelB = B['target']

        # print("length A:", len(A), "length B:", len(B))

        if(len(A) == 0):
            self.leftNode = TreeNode(
                A, LabelA, self.DT, self.depth+1)
            final_predict = list(self.x["target"])
            self.leftNode.redict = max(set(final_predict),
                                       key=final_predict.count)

        elif(len(A) > 0):
            self.leftNode = TreeNode(
                A, LabelA, self.DT, self.depth+1)
            self.leftNode.parent = self

            if(len(A) == len(self.x)) and ((A == self.x).all().all()):
                final_predict = list(self.x["target"])
                self.predict = max(set(final_predict),
                                   key=final_predict.count)
            else:
                self.leftNode.BuildTree()

        if(len(B) == 0):
            self.rightNode = TreeNode(
                B, LabelB, self.DT, self.depth+1)
            final_predict = list(self.x["target"])
            self.rightNode.predict = max(set(final_predict),
                                         key=final_predict.count)

        elif(len(B) > 0):
            self.rightNode = TreeNode(
                B, LabelB, self.DT, self.depth+1)
            self.rightNode.parent = self

            if(len(B) == len(self.x)) and ((B == self.x).all().all()):
                final_predict = list(self.x["target"])
                self.predict = max(set(final_predict),
                                   key=final_predict.count)
            else:
                self.rightNode.BuildTree()

    def Show(self):
        '''
        Node:
            1. leaf: show class( predict ), criterion
            2. have child: show split method, criterion
        '''
        if self.leftNode:
            # have child
            self.leftNode.Show()
            print(self)
            self.rightNode.Show()
        else:
            print(self)

    def __str__(self):
        # show ONLY this node
        to_return = ''
        to_return += '  '*self.depth
        to_return += f'{self.depth}> '
        to_return += '  '*(self.DT.max_depth-self.depth)
        to_return += f'Crit: {self.criterion:<6.3f}'

        if self.leftNode:
            # have child
            # show ONLY this node: threshold
            to_return += f'{self.criterion_attr:>25} - {self.criterion_attr_value:<10}'
        else:
            # leaf node
            to_return += f'VALUE: {self.predict:<3}'

        return to_return


# ### Question 2.1
# Using `criterion=gini`, showing the accuracy score of test data by `max_depth=3` and `max_depth=10`, respectively.
#


# In[8]:

origin_thal = []
[origin_thal.append(i) for i in list(train_df["thal"]) if i not in origin_thal]

train_df = train_df.replace({'fixed': 0, 'reversible': 1, 'normal': 2})
test_df = test_df.replace({'fixed': 0, 'reversible': 1, 'normal': 2})

# train_df = train_df.replace({'fixed': 2, 'reversible': 1, 'normal': 0})
# test_df = test_df.replace({'fixed': 2, 'reversible': 1, 'normal': 0})

# x_train = train_df[list(train_df.columns)[:-1]]
# y_train = train_df[list(train_df.columns)[-1]]

# x_test = test_df[list(test_df.columns)[:-1]]
# y_test = test_df[list(test_df.columns)[-1]]

x_train = train_df.drop("target", axis=1)
y_train = train_df["target"]

x_test = test_df.drop("target", axis=1)
y_test = test_df["target"]

clf_depth3 = DecisionTree(criterion='gini', max_depth=3)
clf_depth10 = DecisionTree(criterion='gini', max_depth=10)

clf_depth3.fit(x_train, y_train)
clf_depth10.fit(x_train, y_train)
# clf_depth3.Show()

print("-------------------------------------------------------------------------------")

y_pred = clf_depth3.predict(x_test)
print("DecisionTree(criterion='gini', max_depth=3), Accuracy Score:",
      accuracy(y_pred, y_test))

y_pred = clf_depth10.predict(x_test)
print("DecisionTree(criterion='gini', max_depth=10), Accuracy Score:",
      accuracy(y_pred, y_test))


# ### Question 2.2
# Using `max_depth=3`, showing the accuracy score of test data by `criterion=gini` and `criterion=entropy`, respectively.
#

# In[9]:

clf_gini = DecisionTree(criterion='gini', max_depth=3)
clf_entropy = DecisionTree(criterion='entropy', max_depth=3)

clf_gini.fit(x_train, y_train)
clf_entropy.fit(x_train, y_train)

print("-------------------------------------------------------------------------------")

y_pred = clf_gini.predict(x_test)
print("DecisionTree(criterion='gini', max_depth=3), Accuracy Score:",
      accuracy(y_pred, list(y_test)))

y_pred = clf_entropy.predict(x_test)
print("DecisionTree(criterion='entropy', max_depth=3), Accuracy Score:",
      accuracy(y_pred, list(y_test)))


# - Note: All of your accuracy scores should over **0.7**
# - Note: You should get the same results when re-building the model with the same arguments,  no need to prune the trees
# - Hint: You can use the recursive method to build the nodes
#

# ## Question 3
# Plot the [feature importance](https://sefiks.com/2020/04/06/feature-importance-in-decision-trees/) of your Decision Tree model. You can get the feature importance by counting the feature used for splitting data.
#
# - You can simply plot the **counts of feature used** for building tree without normalize the importance. Take the figure below as example, outlook feature has been used for splitting for almost 50 times. Therefore, it has the largest importance
#
# ![image](https://i2.wp.com/sefiks.com/wp-content/uploads/2020/04/c45-fi-results.jpg?w=481&ssl=1)

clf_depth10.plot_feature_importance()
print("-------------------------------------------------------------------------------")

print("The feature importance image is saved to feature_importance.png...")


# ## Question 4
# implement the AdaBooest algorithm by using the CART you just implemented from question 2 as base learner. You should implement one arguments for the AdaBooest.
# 1. **n_estimators**: The maximum number of estimators at which boosting is terminated

# In[343]:


class AdaBoost():
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators
        self.total_error = 0
        self.n_total_error = []
        self.amount_value = None
        self.Alldata = None
        self.newAlldata = None
        self.predict_dt = []
        self.sample_weight = None

    def fit(self, x, y):

        self.Alldata = x.copy()
        self.Alldata["target"] = y.copy()
        self.update_Alldata = self.Alldata.copy()

        np.random.seed(12)
        for epochs in range(self.n_estimators):

            # self.Alldata = self.update_Alldata.sample(
            #     n=len(self.update_Alldata), weights=self.sample_weight)
            self.sample_weight = np.full(len(x), 1/len(x))

            if epochs == 0:
                self.newAlldata = self.Alldata.copy()
            else:
                self.newAlldata = self.Alldata.copy()
                self.newAlldata = self.newAlldata.sample(
                    n=len(self.newAlldata), weights=list(self.sample_weight), replace=True, axis=0)

            dt = DecisionTree(max_depth=1)

            dt.fit(self.newAlldata.drop("target", axis=1),
                   self.newAlldata["target"])

            self.target = np.array(self.newAlldata["target"])

            # self.classification = np.full(len(self.target), 1)
            # self.classification[self.Alldata[dt.root.criterion_attr]
            #                     < dt.root.criterion_attr_value] = 0

            self.classification = np.array(dt.predict(
                self.newAlldata.drop("target", axis=1)))

            # print("accuracy:", accuracy(self.classification, self.target))

            misclassification = self.sample_weight[self.target !=
                                                   self.classification]
            self.total_error = np.sum(misclassification)

            self.amount_value = self.calculate_amount_value()

            self.predict_dt.append([dt, self.amount_value])
            self.update_sample_weight()
            # self.update_data()

    def calculate_amount_value(self):

        EPS = 1e-10

        # min_error = self.total_error

        min_error = float('inf')
        if(self.total_error > 0.5):
            self.total_error = 1 - self.total_error
        if(self.total_error < min_error):
            min_error = self.total_error

        return 0.5*(np.log((1-min_error)/(float(min_error)+EPS)))

    def update_sample_weight(self):

        used_signal = np.full(len(self.sample_weight), -1)
        used_signal[self.target != self.classification] = 1

        # calculate new weight
        self.sample_weight = self.sample_weight * \
            np.exp(used_signal*self.amount_value)

        # normalize
        self.sample_weight = self.sample_weight / \
            np.sum(self.sample_weight)

    def update_data(self):

        # random number(0~1)
        # new table is the same as the origin table
        #  0.07     0.07     0.07      0.49     0.07     0.07      0.07      0.07
        # 0~0.07 0.07~0.14 0.14~0.21 0.21~0.7 0.7~0.77 0.77~0.84 0.84~0.91 0.91~0.98

        self.newAlldata = self.Alldata.copy()

        self.newAlldata = self.newAlldata.sample(
            n=len(self.newAlldata), weights=list(self.sample_weight), replace=True)
        self.Alldata = self.newAlldata.copy()

    def predict(self, x):

        clf_predictions = np.array([np.array(dt.predict(x))
                                   for dt, weight in self.predict_dt])

        # predictions = np.sign(clf_predictions.T)
        # final_predictions = [Counter(x).most_common(1)[0][0]
        #                      for x in predictions]
        # predictions = final_predictions

        predictions = []
        for sample_predictions in clf_predictions.T:
            class_0 = 0
            class_1 = 0
            for estimators_idx, predictor_op in enumerate(sample_predictions):
                if predictor_op == 0:
                    class_0 += self.predict_dt[estimators_idx][1]
                else:
                    class_1 += self.predict_dt[estimators_idx][1]

            if class_0 > class_1:
                predictions.append(0)
            else:
                predictions.append(1)

        return predictions

# In[ ]:


# ### Question 4.1
# Show the accuracy score of test data by `n_estimators=10` and `n_estimators=100`, respectively.
#


print("-------------------------------------------------------------------------------")

ad10 = AdaBoost(n_estimators=10)
ad10.fit(x_train, y_train)
y_pred = ad10.predict(x_test)
print("AdaBoost(n_estimators=10), Accuracy Score:", accuracy(y_pred, y_test))

ad100 = AdaBoost(n_estimators=100)
ad100.fit(x_train, y_train)
y_pred = ad100.predict(x_test)
print("AdaBoost(n_estimators=100), Accuracy Score:", accuracy(y_pred, y_test))
# In[ ]:


# ## Question 5
# implement the Random Forest algorithm by using the CART you just implemented from question 2. You should implement three arguments for the Random Forest.
#
# 1. **n_estimators**: The number of trees in the forest.
# 2. **max_features**: The number of random select features to consider when looking for the best split
# 3. **bootstrap**: Whether bootstrap samples are used when building tree
#

# In[11]:

class RandomForest():
    def __init__(self, n_estimators, max_features, boostrap=True, criterion='gini', max_depth=None):
        self.n_estimators = n_estimators
        self.max_features = int(max_features)
        self.boostrap = boostrap
        self.criterion = criterion
        self.max_depth = max_depth
        self.feature_names = None
        self.classifiers = []

    def fit(self, x, y):
        np.random.seed()
        for estimator in range(self.n_estimators):
            self.feature_names = np.array(list(x.columns))

            np.random.shuffle(self.feature_names)
            number_features = self.max_features
            used_feature = self.feature_names[:number_features]

            # selected_feature_idx = random.sample(
            #     range(len(self.feature_names)), self.max_features)

            # selected_feature_idx = sorted(selected_feature_idx)
            # used_feature = self.feature_names[selected_feature_idx]

            dt = DecisionTree(criterion=self.criterion,
                                  max_depth=self.max_depth)

            if self.boostrap:
                boostrap_data = x[used_feature]
                boostrap_data = boostrap_data.copy()

                used_data_idx = np.random.choice(range(len(x)), len(x), replace=True)

                # x_new, y_new = boostrap_data[used_feature], boostrap_data["target"]
                x_new, y_new = boostrap_data.iloc[used_data_idx], y.iloc[used_data_idx]

                dt.fit(x_new, y_new)

                # y_pred = dt.predict(x_new)
                # print("accuracy:", accuracy(y_pred, y_new))

            else:
                normal_data = x[used_feature]
                normal_data = normal_data.copy()

                # x_new, y_new = normal_data[used_feature], normal_data["target"]
                x_new, y_new = normal_data, y

                dt.fit(x_new, y_new)
            self.classifiers.append([dt, used_feature])

        return None

    def predict(self, x):
        predictions = [np.array(dt.predict(x[used_feature]))
                       for dt, used_feature in self.classifiers]
        predictions = np.array(predictions).T
        # predictions = np.sum(predictions, axis=1)
        # print(predictions)

        final_prediction = [Counter(predict).most_common(1)[0][0]
                            for predict in predictions]
        return final_prediction

# ### Question 5.1
# Using `criterion=gini`, `max_depth=None`, `max_features=sqrt(n_features)`, showing the accuracy score of test data by `n_estimators=10` and `n_estimators=100`, respectively.
#

# In[12]:


print("-------------------------------------------------------------------------------")

clf_10tree = RandomForest(criterion="gini",
                          n_estimators=10, max_features=np.sqrt(x_train.shape[1]), boostrap=True)
clf_10tree.fit(x_train, y_train)
y_pred = clf_10tree.predict(x_test)
print("RandomForest(criterion='gini', n_estimators=10, max_features=np.sqrt(n_features), boostrap=True), Accuracy Score:",
      accuracy(y_pred, y_test))

clf_100tree = RandomForest(criterion="gini",
                           n_estimators=100, max_features=np.sqrt(x_train.shape[1]))
clf_100tree.fit(x_train, y_train)
y_pred = clf_100tree.predict(x_test)
print("RandomForest(criterion='gini', n_estimators=100, max_features=np.sqrt(n_features), boostrap=True), Accuracy Score:",
      accuracy(y_pred, y_test))

# In[ ]:

# ### Question 5.2
# Using `criterion=gini`, `max_depth=None`, `n_estimators=10`, showing the accuracy score of test data by `max_features=sqrt(n_features)` and `max_features=n_features`, respectively.
#

# In[13]:

print("-------------------------------------------------------------------------------")

clf_random_features = RandomForest(criterion="gini",
                                   n_estimators=10, max_features=np.sqrt(x_train.shape[1]), boostrap=True)
clf_random_features.fit(x_train, y_train)
y_pred = clf_random_features.predict(x_test)
print("RandomForest(criterion='gini', n_estimators=10, max_features=np.sqrt(n_features), boostrap=True), Accuracy Score:", accuracy(
    y_pred, y_test))

clf_all_features = RandomForest(criterion="gini",
                                n_estimators=10, max_features=x_train.shape[1], boostrap=True)
clf_all_features.fit(x_train, y_train)
y_pred = clf_all_features.predict(x_test)
print("RandomForest(criterion='gini', n_estimators=10, max_features=n_features, boostrap=True), Accuracy Score:",
      accuracy(y_pred, y_test))


# - Note: Use majority votes to get the final prediction, you may get slightly different results when re-building the random forest model

# In[ ]:


# ### Question 6.
# Try you best to get highest test accuracy score by
# - Feature engineering
# - Hyperparameter tuning
# - Implement any other ensemble methods, such as gradient boosting. Please note that you cannot call any package. Also, only ensemble method can be used. Neural network method is not allowed to used.

# In[ ]:


# In[4]:

y_test = test_df['target']


# In[ ]:

max_accuracy = -1
model = None

print("-------------------------------------------------------------------------------")

print("Decision Tree Process")
for dec in range(1, 10):
    dt = DecisionTree(max_depth=dec)
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    value = accuracy(y_pred, y_test)
    if(value > max_accuracy):
        print("DecisionTree(criterion='gini', max_depth=" +
              str(dec)+"), Accuracy Score:", value)
        max_accuracy = value
        model = dt

print("-------------------------------------------------------------------------------")

print("Adaboost Process(default criterion gini)")
for ada_dt in range(1, 30):
    dt = AdaBoost(n_estimators=ada_dt)
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    value = accuracy(y_pred, y_test)
    if(value > max_accuracy):
        print("AdaBoost(n_estimators="+str(ada_dt)+"), Accuracy Score:", value)
        max_accuracy = value
        model = dt

print("-------------------------------------------------------------------------------")

print("Random Forest Process")
for rand_dt in range(1, 10):
    for feature_num in range(2, x_test.shape[1]+1):
        for boostrap in [True, False]:
            dt = RandomForest(criterion="gini",
                              n_estimators=rand_dt, max_depth=4, max_features=feature_num, boostrap=boostrap)
            dt.fit(x_train, y_train)
            y_pred = dt.predict(x_test)
            value = accuracy(y_pred, y_test)
            if(value > max_accuracy):
                print(
                    "RandomForest(criterion='gini', n_estimators="+str(rand_dt)+", max_features="+str(feature_num)+", boostrap="+str(boostrap)+"), Accuracy Score:", value)
                max_accuracy = value
                model = dt


y_pred = model.predict(x_test)

# In[ ]:

print('Test-set accuarcy score: ', accuracy_score(y_test, y_pred))

# ## Supplementary
# If you have trouble to implement this homework, TA strongly recommend watching [this video](https://www.youtube.com/watch?v=LDRbO9a6XPU), which explains Decision Tree model clearly. But don't copy code from any resources, try to finish this homework by yourself!

# In[ ]:
