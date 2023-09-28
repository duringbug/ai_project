'''
Description:
Author: 唐健峰
Date: 2023-09-18 11:13:32
LastEditors: ${author}
LastEditTime: 2023-09-28 14:25:32
'''

from cloud.duringbug.dao.data import *
from cloud.duringbug.preprocessing.index import *
from cloud.duringbug.preprocessing.read import train_file_divide
from cloud.duringbug.train.decision_tree_train import processing_json_txt
from cloud.duringbug.train.train_txt_processing import getX, getW, getL

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import os
import numpy as np
import pdb


def before_decision_tree_to_results_txt():
    train_file_divide()
    dbinit()
    tf_idf_Bow()
    entropy_BoW()
    score_init()
    test_train()


def before_decision_tree_to_results_txt_through_sklearn():
    train_file_divide()
    dbinit()
    tf_idf_Bow()
    entropy_BoW()
    score_init()


def decision_tree_to_results_txt_through_sklearn():
    W = getW()
    with open('resources/exp1_data/my_test.txt', 'r') as file:
        X = getX(file)
    with open('resources/exp1_data/my_test.txt', 'r') as file:
        L = getL(file)
    R = np.dot(X, W)/L
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2'],
        'random_state': [42]
    }
    # 获取训练数据和标签
    # labels(1*4000), result[i]是type*words*score的三维组
    labels, sample, average, result = processing_json_txt(
        "resources/exp1_data/my_train_data.txt", step=5)  # 10分类问题
    labels_verification, sample_verification, average_verification, result_verification = processing_json_txt(
        "resources/exp1_data/my_verification_data.txt", step=5)  # 10分类问题
    # 创建决策树分类器s

    clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
    clf.fit(sample, labels)
    predictions = clf.predict(sample_verification)
    # 比较预测结果与验证标签
    accuracy = accuracy_score(labels_verification, predictions)
    print("Accuracy:", accuracy)
    predictions_R = clf.predict(R)
    with open('out/results.txt', 'w') as file:
        file.write("id, pred\n")
        for row_index, label in enumerate(predictions_R):
            file.write("{} {}\n".format(row_index, label))


def decision_tree_to_results_txt():
    W = getW()
    with open('resources/exp1_data/my_test.txt', 'r') as file:
        X = getX(file)
    R = np.dot(X, W)
    labels, sample, average, result = processing_json_txt(
        "resources/exp1_data/my_train_data.txt", 5)
    threshold_ratio = 0.19
    root_node = RootNode(LeafAverageNode(None, None),
                         LeafCosNode(None, None, average))
    with open('out/results.txt', 'w') as file:
        file.write("id, pred\n")
        for row_index, row in enumerate(R):
            max_col_index = root_node.get_result(
                row.tolist(), threshold_ratio)
            file.write("{} {}\n".format(row_index, max_col_index))


def after_decision_tree_to_results_txt():
    try:
        os.remove("out/results.txt")
        print(f"文件 'out/results.txt' 删除成功")
    except OSError as e:
        print(f"Error deleting file: {e}")

    try:
        os.remove("BoW.db")
        print(f"文件 'BoW.db' 删除成功")
    except OSError as e:
        print(f"Error deleting file: {e}")


def test_train():
    labels, sample, average, result = processing_json_txt(
        "resources/exp1_data/my_train_data.txt", 5)
    labels_2, sample_2, average_2, result_2 = processing_json_txt(
        "resources/exp1_data/my_verification_data.txt", 5)
    # 创建一个RootNode实例，其中true_branch是LeafAverageNode，false_branch是LeafCosNode
    # threshold_ratio = 0.19
    threshold_ratio = 0.19
    root_node = RootNode(LeafAverageNode(None, None),
                         LeafCosNode(None, None, average))
    suc = 0
    all = sample_2.shape[0]
    for column_index in range(sample_2.shape[0]):
        # pdb.set_trace()
        column_list = sample_2[column_index, :].tolist()
        result = root_node.get_result(column_list, threshold_ratio)
        if result == labels[column_index]:
            suc += 1
    print(
        f'threshold_ratio为{threshold_ratio}时,决策树正确率{suc/all},进行余弦算法的个数{root_node.test}')
    root_node.test = 0
    suc = 0
    all = sample_2.shape[0]
    for column_index in range(sample_2.shape[0]):
        # pdb.set_trace()
        column_list = sample_2[column_index, :].tolist()
        result = root_node.get_result(column_list, 1)
        if result == labels[column_index]:
            suc += 1
    print(
        f'threshold_ratio为{0}时,决策树正确率{suc/all},进行余弦算法的个数{root_node.test}')


class RootNode:
    def __init__(self, true_branch, false_branch):
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.test = 0

    def get_result(self, score_matrix_list, threshold_ratio):
        sorted_list = sorted(score_matrix_list, reverse=True)
        sum = 0
        for i in sorted_list:
            sum += i

        if sorted_list[1]/sum < threshold_ratio:
            return self.true_branch.get_result(score_matrix_list)
        else:
            self.test += 1
            return self.false_branch.get_result(score_matrix_list)


class LeafAverageNode:
    def __init__(self, true_branch, false_branch):
        self.true_branch = true_branch
        self.false_branch = false_branch

    def get_result(self, score_matrix_list):
        return max(enumerate(score_matrix_list), key=lambda x: x[1])[0]


class LeafCosNode:
    def __init__(self, true_branch, false_branch, average_matrix):
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.average_matrix = average_matrix

    def get_result(self, score_matrix_list):
        score_vector = np.array(score_matrix_list)
        cosine_distances = np.dot(self.average_matrix, score_vector) / (
            np.linalg.norm(self.average_matrix, axis=1) * np.linalg.norm(score_vector))
        return np.argmax(cosine_distances)
