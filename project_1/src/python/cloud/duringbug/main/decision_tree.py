'''
Description: 
Author: 唐健峰
Date: 2023-09-18 11:13:32
LastEditors: ${author}
LastEditTime: 2023-09-21 11:59:07
'''

from cloud.duringbug.dao.data import *
from cloud.duringbug.preprocessing.index import *
from cloud.duringbug.preprocessing.read import train_file_divide

import numpy as np


def before_decision_tree_to_results_txt():
    train_file_divide()
    dbinit()
    tf_idf_Bow()
    entropy_BoW()
    score_init()


class RootNode:
    def __init__(self, true_branch, false_branch, threshold_ratio):
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.threshold_ratio = threshold_ratio

    def get_result(self, score_matrix_list):
        sorted_list = sorted(score_matrix_list, reverse=True)
        if sorted_list[0]/sorted_list[1] >= self.threshold_ratio:
            return self.true_branch.get_result(score_matrix_list)
        else:
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
