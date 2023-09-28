'''
Description:
Author: 唐健峰
Date: 2023-09-15 18:20:07
LastEditors: ${author}
LastEditTime: 2023-09-28 14:22:40
'''
import sqlite3
import json
import re
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

from cloud.duringbug.preprocessing.read import readPreprocessing
from cloud.duringbug.preprocessing.index import split_text


def train_txt_to_matrix():

    # 创建数据库连接
    conn = sqlite3.connect('BoW.db')
    cursor = conn.cursor()

    cursor.execute('DROP TABLE IF EXISTS train_score')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS train_score (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label_0 INTEGER DEFAULT 0,
        label_1 INTEGER DEFAULT 0,
        label_2 INTEGER DEFAULT 0,
        label_3 INTEGER DEFAULT 0,
        label_4 INTEGER DEFAULT 0,
        label_5 INTEGER DEFAULT 0,
        label_6 INTEGER DEFAULT 0,
        label_7 INTEGER DEFAULT 0,
        label_8 INTEGER DEFAULT 0,
        label_9 INTEGER DEFAULT 0,
        type TEXT,
        forecast TEXT
    )
    ''')
    conn.commit()  # 提交更改

    # 执行SQL查询
    cursor.execute('SELECT * FROM words')

    results = cursor.fetchall()

    # 提取第一列（单词）的数据
    words = [row[0] for row in results]

    cursor.execute('SELECT * FROM score')

    scores = cursor.fetchall()

    # 提取第2至11列的数据
    score_matrix = []

    for row in scores:
        # 假设第2至11列的数据是从索引1到索引10
        row_data = row[1:11]
        score_matrix.append(row_data)

    # 将数据转换为NumPy数组
    score_matrix = np.array(score_matrix)

    data = readPreprocessing("resources/exp1_data/my_train_data.txt")

    # 将字符串分割为多个JSON对象
    json_objects = data.strip().split('\n')

    all_num = len(json_objects)

    one_hot = np.zeros((all_num, 10))
    q_x = np.zeros((all_num, 10))

    recode = 0

    b = np.zeros((all_num, 10))

    for i, json_str in enumerate(tqdm(json_objects, total=all_num, desc="计算偏置向量b中")):

        data = json.loads(json_str)
        label = data["label"]
        raw_text = data["raw"]

        one_hot[i][label] = 1

        # 创建1*N的零矩阵
        sample_matrix = np.zeros((1, len(words)))

        punctuation_to_split = r'[| -,&!".:?();\n$\'#\*-+]+(?!\s)|\s+'
        target_words = split_text(raw_text, punctuation_to_split)

        for target_word in target_words:
            if target_word in words:
                index = words.index(target_word)
                sample_matrix[0][index] += 1

        resalt_matrix = np.dot(sample_matrix, score_matrix)+b[i]
        resalt_matrix = softmax(resalt_matrix)

        q_x[i] = resalt_matrix

        result = minimize(
            my_loss, np.zeros((10, 1)).ravel(), args=(sample_matrix, score_matrix, one_hot[i:i+1, :]), method='BFGS')

        optimized_b = result.x

        b[i] = optimized_b

        cursor.execute(
            'INSERT INTO train_score (label_0,label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8,label_9,type,forecast) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
            (resalt_matrix[0][0], resalt_matrix[0][1], resalt_matrix[0][2], resalt_matrix[0][3], resalt_matrix[0]
             [4], resalt_matrix[0][5], resalt_matrix[0][6], resalt_matrix[0][7], resalt_matrix[0][8], resalt_matrix[0][9], label, np.argmax(resalt_matrix).item())
        )

        if np.argmax(resalt_matrix) == label:
            recode += 1

    conn.commit()  # 提交更改
    # 关闭数据库连接
    conn.close()
    print(f'正确率:{recode/len(json_objects)}')
    mean_values = np.mean(b, axis=0, keepdims=True)
    return mean_values


def test_txt_to_matrix_without_b():

    # 创建数据库连接
    conn = sqlite3.connect('BoW.db')
    cursor = conn.cursor()

    # 执行SQL查询
    cursor.execute('SELECT * FROM words')

    results = cursor.fetchall()

    # 提取第一列（单词）的数据
    words = [row[0] for row in results]

    cursor.execute('SELECT * FROM score')

    scores = cursor.fetchall()

    # 提取第2至11列的数据
    score_matrix = []
    for row in scores:
        # 假设第2至11列的数据是从索引1到索引10
        row_data = row[1:11]
        score_matrix.append(row_data)

    # 将数据转换为NumPy数组
    score_matrix = np.array(score_matrix)

    data = readPreprocessing("resources/exp1_data/my_verification_data.txt")

    # 将字符串分割为多个JSON对象
    json_objects = data.strip().split('\n')

    all_num = len(json_objects)

    one_hot = np.zeros((all_num, 10))
    q_x = np.zeros((all_num, 10))

    recode = 0
    b = np.zeros((1, 10))
    for i, json_str in enumerate(tqdm(json_objects, total=all_num, desc="无偏置向量b下遍历test_my_verification_data中")):

        data = json.loads(json_str)
        label = data["label"]
        raw_text = data["raw"]

        one_hot[i][label] = 1

        # 创建1*N的零矩阵
        sample_matrix = np.zeros((1, len(words)))

        punctuation_to_split = r'[| -,&!".:?();\n$\'#\*-+]+(?!\s)|\s+'
        target_words = split_text(raw_text, punctuation_to_split)

        for target_word in target_words:
            if target_word in words:
                index = words.index(target_word)
                sample_matrix[0][index] += 1

        # 当前实现f(x)=x*W+0
        resalt_matrix = np.dot(sample_matrix, score_matrix)+b
        resalt_matrix = softmax(resalt_matrix)

        q_x[i] = resalt_matrix

        if np.argmax(resalt_matrix) == label:
            recode += 1

    conn.commit()  # 提交更改
    # 关闭数据库连接
    conn.close()
    print(f'正确率:{recode/len(json_objects)}')


def test_txt_to_matrix():

    # 创建数据库连接
    conn = sqlite3.connect('BoW.db')
    cursor = conn.cursor()

    cursor.execute('DROP TABLE IF EXISTS test_score')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS test_score (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label_0 INTEGER DEFAULT 0,
        label_1 INTEGER DEFAULT 0,
        label_2 INTEGER DEFAULT 0,
        label_3 INTEGER DEFAULT 0,
        label_4 INTEGER DEFAULT 0,
        label_5 INTEGER DEFAULT 0,
        label_6 INTEGER DEFAULT 0,
        label_7 INTEGER DEFAULT 0,
        label_8 INTEGER DEFAULT 0,
        label_9 INTEGER DEFAULT 0,
        type TEXT,
        forecast TEXT
    )
    ''')
    conn.commit()  # 提交更改

    # 执行SQL查询
    cursor.execute('SELECT * FROM words')

    results = cursor.fetchall()

    # 提取第一列（单词）的数据
    words = [row[0] for row in results]

    cursor.execute('SELECT * FROM score')

    scores = cursor.fetchall()

    # 提取第2至11列的数据
    score_matrix = []
    for row in scores:
        # 假设第2至11列的数据是从索引1到索引10
        row_data = row[1:11]
        score_matrix.append(row_data)

    # 将数据转换为NumPy数组
    score_matrix = np.array(score_matrix)

    data = readPreprocessing("resources/exp1_data/my_verification_data.txt")

    # 将字符串分割为多个JSON对象
    json_objects = data.strip().split('\n')

    all_num = len(json_objects)

    one_hot = np.zeros((all_num, 10))
    q_x = np.zeros((all_num, 10))

    recode = 0
    b = train_txt_to_matrix()
    for i, json_str in enumerate(tqdm(json_objects, total=all_num, desc="遍历test_my_verification_data中")):

        data = json.loads(json_str)
        label = data["label"]
        raw_text = data["raw"]

        one_hot[i][label] = 1

        # 创建1*N的零矩阵
        sample_matrix = np.zeros((1, len(words)))

        punctuation_to_split = r'[| -,&!".:?();\n$\'#\*-+]+(?!\s)|\s+'
        target_words = split_text(raw_text, punctuation_to_split)

        for target_word in target_words:
            if target_word in words:
                index = words.index(target_word)
                sample_matrix[0][index] += 1

        # 当前实现f(x)=x*W+0
        resalt_matrix = np.dot(sample_matrix, score_matrix)+b
        resalt_matrix = softmax(resalt_matrix)

        q_x[i] = resalt_matrix

        cursor.execute(
            'INSERT INTO test_score (label_0,label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8,label_9,type,forecast) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
            (resalt_matrix[0][0], resalt_matrix[0][1], resalt_matrix[0][2], resalt_matrix[0][3], resalt_matrix[0]
             [4], resalt_matrix[0][5], resalt_matrix[0][6], resalt_matrix[0][7], resalt_matrix[0][8], resalt_matrix[0][9], label, np.argmax(resalt_matrix).item())
        )

        if np.argmax(resalt_matrix) == label:
            recode += 1

    np.savetxt('b.txt', b, fmt='%.16f')
    b_values = np.loadtxt('b.txt')
    print(b_values)

    conn.commit()  # 提交更改
    # 关闭数据库连接
    conn.close()
    print(f'正确率:{recode/len(json_objects)}')


def softmax(matrix):
    softmax_scores = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
    softmax_scores /= np.sum(softmax_scores, axis=1, keepdims=True)
    return softmax_scores
    # 逻辑回归
    # def logistic(X, y, W, b):


def loss(p, q):
    cross_entropy = -np.sum(p * np.log2(q),
                            axis=1, keepdims=True)
    return cross_entropy


def my_loss(b, sample_matrix, score_matrix, one_hot):
    resalt_matrix = np.dot(sample_matrix, score_matrix)+b
    resalt_matrix = softmax(resalt_matrix)
    return loss(one_hot, resalt_matrix)


def getW():
    conn = sqlite3.connect('BoW.db')
    cursor = conn.cursor()
    conn.commit()  # 提交更改

    cursor.execute('SELECT * FROM score')

    scores = cursor.fetchall()

    # 提取第2至11列的数据
    score_matrix = []
    for row in scores:
        # 假设第2至11列的数据是从索引1到索引10
        row_data = row[1:11]
        score_matrix.append(row_data)

    # 将数据转换为NumPy数组
    score_matrix = np.array(score_matrix)
    # N*10
    return score_matrix


def getL(texts):
    text_dict = {}
    for line in texts:
        parts = line.strip().split(', ', 1)
        if len(parts) == 2:
            index, text = parts
            # 将索引转换为整数
            index = int(index)
            # 存储到字典中
            text_dict[index] = text
    L = np.zeros((len(text_dict), 1))
    for i, text in enumerate(tqdm(text_dict, total=len(text_dict), desc="遍历my_test_txt中")):
        punctuation_to_split = r'[| -,&!".:?();\n$\'#\*-+]+(?!\s)|\s+'
        target_words = split_text(text_dict[i], punctuation_to_split)
        L[i][0] = len(target_words)
    return L


def getX(texts):

    # 定义一个空字典来存储文本数据
    text_dict = {}

    for line in texts:
        parts = line.strip().split(', ', 1)
        if len(parts) == 2:
            index, text = parts
            # 将索引转换为整数
            index = int(index)
            # 存储到字典中
            text_dict[index] = text

    # 创建数据库连接
    conn = sqlite3.connect('BoW.db')
    cursor = conn.cursor()

    cursor.execute('DROP TABLE IF EXISTS test_score')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS test_score (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label_0 INTEGER DEFAULT 0,
        label_1 INTEGER DEFAULT 0,
        label_2 INTEGER DEFAULT 0,
        label_3 INTEGER DEFAULT 0,
        label_4 INTEGER DEFAULT 0,
        label_5 INTEGER DEFAULT 0,
        label_6 INTEGER DEFAULT 0,
        label_7 INTEGER DEFAULT 0,
        label_8 INTEGER DEFAULT 0,
        label_9 INTEGER DEFAULT 0,
        type TEXT,
        forecast TEXT
    )
    ''')
    conn.commit()  # 提交更改

    # 执行SQL查询
    cursor.execute('SELECT * FROM words')

    results = cursor.fetchall()

    # 提取第一列（单词）的数据
    words = [row[0] for row in results]

    def split_text(text, punctuation_to_split=None):
        # 默认的标点符号分隔符是空格
        if punctuation_to_split is None:
            punctuation_to_split = r'\s+'

        # 使用正则表达式进行分割
        words = re.split(punctuation_to_split, text.lower())
        return words

    # 创建4000*N的零矩阵
    sample_matrix = np.zeros((len(text_dict), len(words)))
    for i, text in enumerate(tqdm(text_dict, total=len(text_dict), desc="遍历my_test_txt中")):

        punctuation_to_split = r'[| -,&!".:?();\n$\'#\*-+]+(?!\s)|\s+'
        target_words = split_text(text_dict[i], punctuation_to_split)

        for target_word in target_words:
            if target_word in words:
                index = words.index(target_word)
                sample_matrix[i][index] += 1

    # 4000*N
    return sample_matrix


def getb():
    return np.loadtxt('b.txt')
