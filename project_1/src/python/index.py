'''
Description: 
Author: 唐健峰
Date: 2023-09-14 10:13:45
LastEditors: ${author}
LastEditTime: 2023-09-26 21:20:56
'''
from cloud.duringbug.main.logistic_regression import *
from cloud.duringbug.main.decision_tree import *
from cloud.duringbug.main.support_vector_machine import *
from cloud.duringbug.test.check_environment import *


def main():
    return 0


if __name__ == "__main__":
    test_import()
    user_input = ""
    while user_input != "exit":
        print("请输入测试的模型:\n输入1:词袋模型+TDIDF+逻辑回归\n输入2:词袋模型+TDIDF+决策树\n输入3:词袋模型+支持向量机+信息熵\n输入exit:退出")
        user_input = input("")
        if user_input == "1":
            print("预计6min40s")
            before_logistic_regression_to_results_txt()
            logistic_regression_to_results_txt()
            print("回车清理缓存")
            user_input = input("")
            if user_input == "":
                after_logistic_regression_to_results_txt()
        if user_input == "2":
            print("预计1min05s")
            before_decision_tree_to_results_txt()
            decision_tree_to_results_txt()
            print("回车清理缓存")
            user_input = input("")
            if user_input == "":
                after_decision_tree_to_results_txt()
        if user_input == "3":
            print("预计7min02s")
            before_support_vector_machine()
            support_vector_machine()
            print("回车清理缓存")
            user_input = input("")
            if user_input == "":
                after_support_vector_machine()
