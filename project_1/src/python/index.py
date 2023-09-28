'''
Description: 
Author: 唐健峰
Date: 2023-09-14 10:13:45
LastEditors: ${author}
LastEditTime: 2023-09-28 14:28:58
'''
from cloud.duringbug.main.logistic_regression import *
from cloud.duringbug.main.decision_tree import *
from cloud.duringbug.main.support_vector_machine import *
from cloud.duringbug.main.multilayer_perceptron import *
from cloud.duringbug.test.check_environment import *


def main():
    return 0


if __name__ == "__main__":
    test_import()
    user_input = ""
    while user_input != "exit":
        print("请输入测试的模型:\n\
                输入1:词袋模型+TDIDF+逻辑回归\n\
                输入2:词袋模型+TDIDF+决策树\n\
                输入3:词袋模型+信息熵+支持向量机\n\
                输入4:词袋模型+信息熵+mlp\n\
                输入5:词袋模型+TDIDF+sklearn决策树\n\
                输入exit:退出")
        user_input = input("")
        if user_input == "1":
            print("预计6min40s")
            before_logistic_regression_to_results_txt()
            logistic_regression_to_results_txt()
            print("回车清理缓存")
            user_input = input("")
            if user_input == "":
                after_logistic_regression_to_results_txt()
        elif user_input == "2":
            print("预计1min05s")
            before_decision_tree_to_results_txt()
            decision_tree_to_results_txt()
            print("回车清理缓存")
            user_input = input("")
            if user_input == "":
                after_decision_tree_to_results_txt()
        elif user_input == "3":
            print("预计7min02s")
            before_support_vector_machine()
            support_vector_machine()
            print("回车清理缓存")
            user_input = input("")
            if user_input == "":
                after_support_vector_machine()
        elif user_input == "4":
            print("预计1min08s")
            before_multilayer_perceptron()
            multilayer_perceptron()
            print("回车清理缓存")
            user_input = input("")
            if user_input == "":
                after_multilayer_perceptron()
        elif user_input == "5":
            print("预计1min08s")
            before_decision_tree_to_results_txt_through_sklearn()
            decision_tree_to_results_txt_through_sklearn()
            print("回车清理缓存")
            user_input = input("")
            if user_input == "":
                after_decision_tree_to_results_txt()
