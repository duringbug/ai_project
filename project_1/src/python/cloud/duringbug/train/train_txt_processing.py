'''
Description:
Author: 唐健峰
Date: 2023-09-15 18:20:07
LastEditors: ${author}
LastEditTime: 2023-09-16 00:05:14
'''
import sqlite3
import json
import re
import numpy as np
from tqdm import tqdm

from cloud.duringbug.preprocessing.read import readPreprocessing


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
        type TEXT
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

    def split_text(text, punctuation_to_split=None):
        # 默认的标点符号分隔符是空格
        if punctuation_to_split is None:
            punctuation_to_split = r'\s+'

        # 使用正则表达式进行分割
        words = re.split(punctuation_to_split, text.lower())
        return words

    recode = 0
    for i, json_str in enumerate(tqdm(json_objects, total=all_num, desc="遍历test_my_train_txt中")):

        data = json.loads(json_str)
        label = data["label"]
        raw_text = data["raw"]

        # 创建1*N的零矩阵
        sample_matrix = np.zeros((1, len(words)))

        punctuation_to_split = r'[ -,&!".:?();\n$\'#\*-+]'
        target_words = split_text(raw_text, punctuation_to_split)

        for target_word in target_words:
            if target_word in words:
                index = words.index(target_word)
                sample_matrix[0][index] += 1

        resalt_matrix = np.dot(sample_matrix, score_matrix)

        cursor.execute(
            'INSERT INTO train_score (label_0,label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8,label_9,type) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
            (resalt_matrix[0][0], resalt_matrix[0][1], resalt_matrix[0][2], resalt_matrix[0][3], resalt_matrix[0]
             [4], resalt_matrix[0][5], resalt_matrix[0][6], resalt_matrix[0][7], resalt_matrix[0][8], resalt_matrix[0][9], label)
        )

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
        type TEXT
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

    def split_text(text, punctuation_to_split=None):
        # 默认的标点符号分隔符是空格
        if punctuation_to_split is None:
            punctuation_to_split = r'\s+'

        # 使用正则表达式进行分割
        words = re.split(punctuation_to_split, text.lower())
        return words

    recode = 0
    for i, json_str in enumerate(tqdm(json_objects, total=all_num, desc="遍历test_my_verification_data中")):

        data = json.loads(json_str)
        label = data["label"]
        raw_text = data["raw"]

        # 创建1*N的零矩阵
        sample_matrix = np.zeros((1, len(words)))

        punctuation_to_split = r'[ -,&!".:?();\n$\'#\*-+]'
        target_words = split_text(raw_text, punctuation_to_split)

        for target_word in target_words:
            if target_word in words:
                index = words.index(target_word)
                sample_matrix[0][index] += 1

        resalt_matrix = np.dot(sample_matrix, score_matrix)

        cursor.execute(
            'INSERT INTO test_score (label_0,label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8,label_9,type) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
            (resalt_matrix[0][0], resalt_matrix[0][1], resalt_matrix[0][2], resalt_matrix[0][3], resalt_matrix[0]
             [4], resalt_matrix[0][5], resalt_matrix[0][6], resalt_matrix[0][7], resalt_matrix[0][8], resalt_matrix[0][9], label)
        )

        if np.argmax(resalt_matrix) == label:
            recode += 1

    conn.commit()  # 提交更改
    # 关闭数据库连接
    conn.close()
    print(f'正确率:{recode/len(json_objects)}')
