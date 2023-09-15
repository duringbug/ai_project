'''
Description: 
Author: 唐健峰
Date: 2023-09-14 10:55:55
LastEditors: ${author}
LastEditTime: 2023-09-15 15:57:59
'''
import unittest
from cloud.duringbug.dao.data import *

from cloud.duringbug.preprocessing.index import main

import spacy
nlp = spacy.load("en_core_web_sm")


class MyTestCase(unittest.TestCase):

    # def test_spacy(self):
    #     text = "This is an example sentence."
    #     doc = nlp(text)
    #     words = [token.text for token in doc]
    #     print(words)

    # def test_insert(self):
    #     init()
    #     insert("唐健峰")
    #     delete("唐健峰")

    # def test_insertText(self):
    #     main()

    # def test_softmax(self):
    #     entropy_BoW()

    # def test_TF_IDF(self):
    #     TF_IDF()

    # def test_del_database(self):
    #     del_database("TF-IDF.db")

    def test_1(self):
        init()
        main()
        entropy_BoW()
        score_init()

    # def test_2(self):
    #     score_init()


if __name__ == '__main__':
    unittest.main()
