'''
Description: 
Author: 唐健峰
Date: 2023-09-14 10:55:55
LastEditors: ${author}
LastEditTime: 2023-09-16 00:23:02
'''
import unittest
from cloud.duringbug.dao.data import *

from cloud.duringbug.preprocessing.index import main

from cloud.duringbug.train.train_txt_processing import *

from cloud.duringbug.preprocessing.read import train_file_divide

import spacy
nlp = spacy.load("en_core_web_sm")


class MyTestCase(unittest.TestCase):

    def test_divide_text(self):
        train_file_divide()

    def test_train_data_to_vector(self):
        init()
        main()
        entropy_BoW()
        score_init()
        train_txt_to_matrix()
        test_txt_to_matrix()


if __name__ == '__main__':
    unittest.main()
