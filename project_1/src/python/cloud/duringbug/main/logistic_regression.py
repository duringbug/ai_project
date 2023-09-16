from cloud.duringbug.dao.data import *
from cloud.duringbug.preprocessing.index import *
from cloud.duringbug.train.train_txt_processing import *
from cloud.duringbug.preprocessing.read import train_file_divide
from cloud.duringbug.train.train_txt_processing import *

import numpy as np

import sqlite3
import os


def before_logistic_regression_to_results_txt():
    train_file_divide()
    dbinit()
    tf_idf_Bow()
    entropy_BoW()
    score_init()
    test_txt_to_matrix()
    test_txt_to_matrix_without_b()


def logistic_regression_to_results_txt():
    # 加载从文本文件中读取的数组
    W = getW()
    with open('resources/exp1_data/my_test.txt', 'r') as file:
        X = getX(file)
    b = getb()
    with open('out/results.txt', 'w') as file:
        file.write("id, pred\n")
        for row_index, row in enumerate(np.dot(X, W)+b):
            # 找到每行的最大值和对应的列位置
            max_value = max(row)
            max_col_index = np.argmax(row)
            file.write("{} {}\n".format(row_index, max_col_index))


def after_logistic_regression_to_results_txt():
    try:
        os.remove("out/results.txt")
        print(f"文件 'out/results.txt' 删除成功")
    except OSError as e:
        print(f"Error deleting file: {e}")

    try:
        os.remove("b.txt")
        print(f"文件 'b.txt' 删除成功")
    except OSError as e:
        print(f"Error deleting file: {e}")

    try:
        os.remove("BoW.db")
        print(f"文件 'BoW.db' 删除成功")
    except OSError as e:
        print(f"Error deleting file: {e}")
