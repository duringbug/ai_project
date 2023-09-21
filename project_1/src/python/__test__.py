'''
Description:
Author: 唐健峰
Date: 2023-09-14 10:55:55
LastEditors: ${author}
LastEditTime: 2023-09-21 11:42:02
'''
from cloud.duringbug.main.logistic_regression import *
from cloud.duringbug.main.decision_tree import *
from cloud.duringbug.train.decision_tree_train import *

import unittest


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

    def test_root_node(self):
        before_decision_tree_to_results_txt()
        labels, sample, average, result = processing_json_txt(
            "resources/exp1_data/my_train_data.txt", 5)
        labels_2, sample_2, average_2, result_2 = processing_json_txt(
            "resources/exp1_data/my_verification_data.txt", 5)
        # 创建一个RootNode实例，其中true_branch是LeafAverageNode，false_branch是LeafCosNode
        threshold_ratio = 2.0
        root_node = RootNode(LeafAverageNode(None, None),
                             LeafCosNode(None, None, average), threshold_ratio)
        result = root_node.get_result([0.0001480641903602, 0.0000914138233921, 0.0000778772398946, 0.0000900243386887, 0.0000760588464261,
                                       0.0000924187746548, 0.0001037625882078, 0.0000746589811823, 0.0000986877171409, 0.0000989021748436])
        print(result)


if __name__ == '__main__':
    unittest.main()
