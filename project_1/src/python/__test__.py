'''
Description: 
Author: 唐健峰
Date: 2023-09-14 10:55:55
LastEditors: ${author}
LastEditTime: 2023-09-16 20:19:04
'''


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


if __name__ == '__main__':
    unittest.main()
