'''
Description:
Author: 唐健峰
Date: 2023-09-14 10:55:55
LastEditors: ${author}
LastEditTime: 2023-09-28 14:29:48
'''
from cloud.duringbug.test.check_environment import test_import as my_test_import

from cloud.duringbug.main.logistic_regression import *
from cloud.duringbug.main.decision_tree import *
from cloud.duringbug.main.multilayer_perceptron import *
from cloud.duringbug.main.support_vector_machine import *

from cloud.duringbug.train.decision_tree_train import *
from cloud.duringbug.train.multilayer_perceptron_train import *
from cloud.duringbug.train.support_vector_machine_train import *

import unittest
import pdb


class MyTestCase(unittest.TestCase):

    def test_import(self):
        my_test_import()

    def test_clear(self):
        user_input = input("")
        if user_input == "":
            after_logistic_regression_to_results_txt()

    @unittest.skip("跳过这个test_decision_tree方法")
    def test_decision_tree(self):
        before_decision_tree_to_results_txt()
        train_txt_to_decision_tree_sample()

    @unittest.skip("跳过这个test_root_node方法")
    def test_root_node_2(self):
        before_decision_tree_to_results_txt_through_sklearn()
        decision_tree_to_results_txt_through_sklearn()

    @unittest.skip("跳过这个test_root_node方法")
    def test_root_node_1(self):
        before_decision_tree_to_results_txt()
        decision_tree_to_results_txt()

    @unittest.skip("跳过这个test_support_vector_machine_train方法")
    def test_support_vector_machine_train(self):
        before_support_vector_machine()
        support_vector_machine()

    @unittest.skip("跳过这个test_multilayer_perceptron方法")
    def test_multilayer_perceptron(self):
        before_multilayer_perceptron()
        multilayer_perceptron()


if __name__ == '__main__':
    unittest.main()
