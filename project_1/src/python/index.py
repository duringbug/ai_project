'''
Description: 
Author: 唐健峰
Date: 2023-09-14 10:13:45
LastEditors: ${author}
LastEditTime: 2023-09-16 20:57:34
'''
from cloud.duringbug.main.logistic_regression import *
from cloud.duringbug.test.check_environment import *


def main():
    return 0


if __name__ == "__main__":
    test_import()
    user_input = ""
    while user_input != "exit":
        print("请输入测试的模型:\n输入1:词袋模型+TDIDF+逻辑回归\n输入exit:退出")
        user_input = input("")
        if user_input == "1":
            print("预计6min40s")
            before_logistic_regression_to_results_txt()
            logistic_regression_to_results_txt()
            print("回车清理缓存")
            user_input = input("")
            if user_input:
                after_logistic_regression_to_results_txt()
