'''
Description:
Author: 唐健峰
Date: 2023-09-14 10:23:00
LastEditors: ${author}
LastEditTime: 2023-09-16 19:26:14
'''

import json
import math
import re
import sqlite3
from tqdm import tqdm
from collections import defaultdict
from cloud.duringbug.preprocessing.read import readPreprocessing
from cloud.duringbug.dao.data import insertText
from cloud.duringbug.dao.data import TF_IDF


def tf_idf_Bow():

    data = readPreprocessing("resources/exp1_data/my_train_data.txt")

    # 将字符串分割为多个JSON对象
    json_objects = data.strip().split('\n')

    # 创建一个字典用于存储每个词语的文档频率
    document_frequency = defaultdict(int)

    def split_text(text, punctuation_to_split=None):
        # 默认的标点符号分隔符是空格
        if punctuation_to_split is None:
            punctuation_to_split = r'\s+'

        # 使用正则表达式进行分割
        words = re.split(punctuation_to_split, text.lower())
        return words

    # 解析每个JSON对象
    all_num = len(json_objects)
    # 创建包含十个字典的数组
    ten_dicts = [defaultdict(int) for _ in range(10)]
    ten_idf_values = [{} for _ in range(10)]

    for json_str in tqdm(json_objects, total=all_num, desc="遍历my_train_txt中"):

        data = json.loads(json_str)
        label = data["label"]
        raw_text = data["raw"]

        # 将文本转换为小写并拆分为单词
        punctuation_to_split = r'[ -,&!".:?();\n$\'#\*-+]'
        words = split_text(raw_text, punctuation_to_split)

        for word in set(words):  # 使用 set 去除重复词语
            ten_dicts[label][word] += 1
        # insertText函数包括储存数据
        insertText(words, label)

    # 计算IDF
    for i, idf_values in enumerate(ten_idf_values):
        # @ TODO 数据划分时记得改
        total_documents = 400
        all_num_2 = len(ten_dicts[i])
        for word, df in tqdm(ten_dicts[i].items(), total=all_num_2, desc=f'计算第{i}类文章的IDF'):
            # 使用对数形式计算IDF，避免分母为0
            idf = math.log(total_documents / (df + 1))
            idf_values[word] = idf

    # 计算频率，返回10个TF字典数组
    ten_tf_values = TF_IDF()

    # 连接TF-IDF
    conn = sqlite3.connect('BoW.db')
    cursor = conn.cursor()

    # 计算ten_tf_values*ten_idf_values

    TF_IDF_results = [{} for _ in range(10)]

    for i in range(10):
        for key in ten_tf_values[i].keys():
            if key in ten_idf_values[i]:
                TF_IDF_results[i][key] = ten_tf_values[i][key] * \
                    ten_idf_values[i][key]

    for key in tqdm(ten_tf_values[i].keys(), total=len(ten_tf_values[i]), desc=f'储存TF-IDF中'):
        cursor.execute(
            'INSERT INTO TF_IDF (word,label_0,label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8,label_9) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
            (key, TF_IDF_results[0].get(key, 0), TF_IDF_results[1].get(key, 0), TF_IDF_results[2].get(key, 0), TF_IDF_results[3].get(key, 0), TF_IDF_results[4].get(
                key, 0), TF_IDF_results[5].get(key, 0), TF_IDF_results[6].get(key, 0), TF_IDF_results[7].get(key, 0), TF_IDF_results[8].get(key, 0), TF_IDF_results[9].get(key, 0))
        )

    conn.commit()
    # 提交更改并关闭数据库连接
    conn.close()
