'''
Description: 
Author: 唐健峰
Date: 2023-09-14 10:36:58
LastEditors: ${author}
LastEditTime: 2023-09-26 14:50:04
'''

import sqlite3


def get_sorce_dict():
    conn = sqlite3.connect('BoW.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM score')
    scores = cursor.fetchall()
    conn.close()  # 关闭数据库连接

    score_data = {}
    for row in scores:
        word = row[0]  # 假设第1列是学生ID
        scores_list = row[1:11]  # 假设第2至11列是成绩数据
        score_data[word] = scores_list

    return score_data


def get_words_zero_dict():
    conn = sqlite3.connect('BoW.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM words')
    results = cursor.fetchall()
    conn.close()  # 关闭数据库连接
    words = {}
    for row in results:
        words[row[0]] = 0
    return words
