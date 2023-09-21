'''
Description: 
Author: 唐健峰
Date: 2023-09-14 10:36:58
LastEditors: ${author}
LastEditTime: 2023-09-18 17:23:32
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
