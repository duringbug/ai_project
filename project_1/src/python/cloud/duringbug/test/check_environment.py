'''
Description:
Author: 唐健峰
Date: 2023-09-16 20:22:06
LastEditors: ${author}
LastEditTime: 2023-09-26 18:47:50
'''


def test_import():
    try:
        from sklearn.metrics import accuracy_score
        from sklearn.svm import SVC
        import unittest
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
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        print("所有所需的包都已安装并可用。")
    except ImportError as e:
        missing_module = str(e).split()[-1]
        print(f"缺少以下包: {missing_module}")
