'''
Description:
Author: 唐健峰
Date: 2023-09-14 10:55:55
LastEditors: ${author}
LastEditTime: 2023-09-21 17:40:39
'''
from cloud.duringbug.main.logistic_regression import *
from cloud.duringbug.main.decision_tree import *
from cloud.duringbug.train.decision_tree_train import *

import unittest
import pdb


class MyTestCase(unittest.TestCase):

    def test_import(self):
        try:
            import numpy
            from tqdm import tqdm
            import sqlite3
            import os
            import re
            import json
            import math
            from collections import defaultdict
            import random
            from scipy.optimize import minimize
            print("所有所需的包都已安装并可用。")
        except ImportError as e:
            missing_module = str(e).split()[-1]
            print(f"缺少以下包: {missing_module}")

    def test_clear(self):
        user_input = input("")
        if user_input == "":
            after_logistic_regression_to_results_txt()

    @unittest.skip("跳过这个test_decision_tree方法")
    def test_decision_tree(self):
        before_decision_tree_to_results_txt()
        train_txt_to_decision_tree_sample()

    # @unittest.skip("跳过这个test_root_node方法")
    def test_root_node(self):
        before_decision_tree_to_results_txt()
        labels, sample, average, result = processing_json_txt(
            "resources/exp1_data/my_train_data.txt", 5)
        labels_2, sample_2, average_2, result_2 = processing_json_txt(
            "resources/exp1_data/my_verification_data.txt", 5)
        # 创建一个RootNode实例，其中true_branch是LeafAverageNode，false_branch是LeafCosNode
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


if __name__ == '__main__':
    unittest.main()
