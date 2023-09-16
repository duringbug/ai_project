'''
Description: 
Author: 唐健峰
Date: 2023-09-14 10:55:55
LastEditors: ${author}
LastEditTime: 2023-09-16 15:32:49
'''


import unittest
from cloud.duringbug.dao.data import *
from cloud.duringbug.preprocessing.index import main
from cloud.duringbug.train.train_txt_processing import *
from cloud.duringbug.preprocessing.read import train_file_divide

from cloud.duringbug.main.logistic_regression import index


class MyTestCase(unittest.TestCase):

    def test_logistic_regression(self):
        index()


if __name__ == '__main__':
    unittest.main()
