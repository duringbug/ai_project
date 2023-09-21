'''
Description:
Author: 唐健峰
Date: 2023-09-14 10:35:44
LastEditors: ${author}
LastEditTime: 2023-09-18 17:21:15
'''
import numpy as np
from tqdm import tqdm
import sqlite3
import os
import re


def dbinit():
    if os.path.exists('BoW.db'):
        os.remove('BoW.db')
    conn = sqlite3.connect('BoW.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS words (
        word TEXT PRIMARY KEY,
        label_0 INTEGER DEFAULT 0,
        label_1 INTEGER DEFAULT 0,
        label_2 INTEGER DEFAULT 0,
        label_3 INTEGER DEFAULT 0,
        label_4 INTEGER DEFAULT 0,
        label_5 INTEGER DEFAULT 0,
        label_6 INTEGER DEFAULT 0,
        label_7 INTEGER DEFAULT 0,
        label_8 INTEGER DEFAULT 0,
        label_9 INTEGER DEFAULT 0
    )
''')
    conn.commit()  # 提交更改
    conn.close()   # 关闭数据库连接


def insert(word):
    conn = sqlite3.connect('BoW.db')
    cursor = conn.cursor()

    try:
        cursor.execute('INSERT INTO words (word) VALUES (?)', (word,))
        conn.commit()  # 提交更改
        print(f"Insert word: {word}")
    except sqlite3.Error as e:
        print(f"Error insert word: {e}")
    finally:
        # 关闭数据库连接
        conn.close()


def del_database(database_file):
    try:
        # 检查数据库文件是否存在
        if os.path.exists(database_file):
            # 删除数据库文件
            os.remove(database_file)
            print(f"数据库文件 '{database_file}' 已成功删除。")
        else:
            print(f"数据库文件 '{database_file}' 不存在，无需删除。")
    except Exception as e:
        print(f"删除数据库文件时发生错误：{e}")


def delete(word):
    # 创建数据库连接
    conn = sqlite3.connect('BoW.db')
    cursor = conn.cursor()

    try:
        # 执行删除操作
        cursor.execute('DELETE FROM words WHERE word = ?', (word,))

        # 提交更改
        conn.commit()

        print(f"Deleted word: {word}")
    except sqlite3.Error as e:
        print(f"Error deleting word: {e}")
    finally:
        # 关闭数据库连接
        conn.close()


def insertText(words, label):
    # 连接到数据库
    conn = sqlite3.connect('BoW.db')
    cursor = conn.cursor()

    # 对每个单词进行处理
    for word in words:
        # 检查数据库中是否已存在该单词
        cursor.execute('SELECT * FROM words WHERE word = ?', (word,))
        existing_word = cursor.fetchone()

        if existing_word:
            # 如果单词已存在，更新相应的 label_ 列
            update_query = f'UPDATE words SET label_{label} = label_{label} + 1 WHERE word = ?'
            cursor.execute(update_query, (word,))
        else:
            # 如果单词不存在，创建它并设置相应的 label_ 列
            insert_query = f'INSERT INTO words (word, label_{label}) VALUES (?, 1)'
            cursor.execute(insert_query, (word,))

    # 提交更改并关闭数据库连接
    conn.commit()
    conn.close()


def entropy_BoW():
    # 连接到数据库
    conn = sqlite3.connect('BoW.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS entropy (
        word TEXT PRIMARY KEY,
        entropy_value REAL DEFAULT 0.0
    )
''')
    conn.commit()  # 提交更改

    # 执行SQL查询
    cursor.execute('SELECT * FROM words')

    results = cursor.fetchall()

    # 循环输出查询结果
    all_num = len(results)

    def calculate_entropy(probabilities):
        entropy = 0.0
        for p in probabilities:
            if p != 0:
                entropy -= p * np.log2(p)
        return np.e ** entropy

    for row in tqdm(results, total=all_num, desc="计算每个词的信息熵"):
        # row 是一个包含查询结果的元组，您可以根据需要访问其中的列数据
        # 例如，如果有两列：id 和 content，可以使用 row[0] 访问 id，row[1] 访问 content
        scores = np.array([row[1], row[2], row[3], row[4],
                           row[5], row[6], row[7], row[8], row[9], row[10]])
        probabilities = scores / np.sum(scores)
        entropy = calculate_entropy(probabilities)
        cursor.execute(
            'INSERT INTO entropy (word, entropy_value) VALUES (?, ?)',
            (row[0], entropy)
        )

    conn.commit()
    # 关闭数据库连接
    conn.close()


def TF_IDF():
    # 连接到数据库
    conn = sqlite3.connect('BoW.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS TF_IDF (
        word TEXT PRIMARY KEY,
        label_0 INTEGER DEFAULT 0,
        label_1 INTEGER DEFAULT 0,
        label_2 INTEGER DEFAULT 0,
        label_3 INTEGER DEFAULT 0,
        label_4 INTEGER DEFAULT 0,
        label_5 INTEGER DEFAULT 0,
        label_6 INTEGER DEFAULT 0,
        label_7 INTEGER DEFAULT 0,
        label_8 INTEGER DEFAULT 0,
        label_9 INTEGER DEFAULT 0
    )
''')
    conn.commit()  # 提交更改
    cursor.execute('SELECT * FROM words')

    results = cursor.fetchall()

    # 循环输出查询结果
    all_num = len(results)

    all_word_per_label_matrix = np.zeros((1, 10))

    ten_TF_dicts = [{} for _ in range(10)]

    for row in tqdm(results, total=all_num, desc="计算每类的词总数"):
        all_word_per_label_matrix[0, :] += row[1:11]
    for j, row in enumerate(tqdm(results, total=all_num, desc="计算TF中")):
        for i in range(1, 11):
            ten_TF_dicts[i-1][row[0]] = row[i] / \
                (all_word_per_label_matrix[0][i-1])
    conn.close()
    return ten_TF_dicts


def score_init():
    conn = sqlite3.connect('BoW.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS score (
        word TEXT PRIMARY KEY,
        label_0 NUMERIC DEFAULT 0,
        label_1 NUMERIC DEFAULT 0,
        label_2 NUMERIC DEFAULT 0,
        label_3 NUMERIC DEFAULT 0,
        label_4 NUMERIC DEFAULT 0,
        label_5 NUMERIC DEFAULT 0,
        label_6 NUMERIC DEFAULT 0,
        label_7 NUMERIC DEFAULT 0,
        label_8 NUMERIC DEFAULT 0,
        label_9 NUMERIC DEFAULT 0
    )
''')
    conn.commit()  # 提交更改
    # 获取entropy数据，假设entropy是N*1的矩阵
    cursor.execute('SELECT entropy_value FROM entropy')
    entropy_data = np.array(cursor.fetchall())

    # 获取TF-IDF数据，假设TF-IDF是N*10的矩阵
    cursor.execute(
        'SELECT label_0, label_1, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_9 FROM TF_IDF')
    TF_IDF_data = np.array(cursor.fetchall())

    # 执行矩阵乘法(N*10)/(N*1)
    result = TF_IDF_data.astype(float) / entropy_data.astype(float)

    cursor.execute('SELECT * FROM words')

    rows = cursor.fetchall()

    for index, row in enumerate(tqdm(rows, total=len(rows), desc="储存score向量矩阵")):
        cursor.execute(
            'INSERT INTO score (word,label_0,label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8,label_9) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
            (row[0], result[index][0], result[index][1], result[index][2], result[index][3], result[index]
             [4], result[index][5], result[index][6], result[index][7], result[index][8], result[index][9])
        )

    conn.commit()  # 提交更改
    conn.close()  # 关闭数据库连接
