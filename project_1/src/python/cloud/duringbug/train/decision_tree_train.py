'''
Description:
Author: 唐健峰
Date: 2023-09-18 12:54:42
LastEditors: ${author}
LastEditTime: 2023-09-26 13:20:28
'''
# -- coding: utf-8 --

from cloud.duringbug.preprocessing.read import readPreprocessing
from cloud.duringbug.preprocessing.index import split_text
from cloud.duringbug.dao.index import get_sorce_dict
import sqlite3
from tqdm import tqdm
import json
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cloud.duringbug.conf.config import AppConfig

app_config = AppConfig()
app_config.load_config_from_file('resources/config/config.json')


def train_txt_to_decision_tree_sample():
    labels, sample, average, result = processing_json_txt(
        "resources/exp1_data/my_train_data.txt", 5)
    labels_2, sample_2, average_2, result_2 = processing_json_txt(
        "resources/exp1_data/my_verification_data.txt", 5)

    dis = cos_dis(sample, average, labels)
    average_dis(sample, labels)

    cos_dis(sample_2, average, labels_2)
    average_dis(sample_2, labels_2)

    np.savetxt('dis.txt', dis, fmt='%.16f')
    np.savetxt('sample.txt', sample, fmt='%.16f')
    create3D(average)
    for k in range(2401, 2801):
        create_surf(k, result)


def processing_json_txt(path, step):
    score_dict = get_sorce_dict()
    data = readPreprocessing(path)
    # 将字符串分割为多个JSON对象
    json_objects = data.strip().split('\n')

    all_num = len(json_objects)
    result = [[] for _ in range(all_num)]
    average = np.zeros((10, 10))
    sample = np.zeros((len(json_objects), 10))
    labels = [0] * sample.shape[0]
    for i, json_str in enumerate(tqdm(json_objects, total=all_num, desc=f'遍历{path}中')):
        data = json.loads(json_str)
        label = data["label"]
        raw_text = data["raw"]
        labels[i] = label
        punctuation_to_split = r'[| -,&!".:?();\n$\'#\*-+]+(?!\s)|\s+'
        target_words = split_text(raw_text, punctuation_to_split)

        # 创建一个全是0的列表，长度等于target_words
        result_row = [np.zeros(10) for _ in range(len(target_words) - step)]
        for j, target_word in enumerate(target_words):
            average[label, :] += score_dict.get(target_words[j], np.zeros(10))
            sample[i, :] += score_dict.get(target_words[j], np.zeros(10))
            if (j + step < len(target_words)):
                for k in range(step):
                    # 使用get()获取值，如果不存在则默认为0
                    result_row[j] += score_dict.get(
                        target_words[j + k], np.zeros(10))
        sample[i, :] = sample[i, :]/len(target_words)
        result[i] = result_row  # 添加result_row到result列表

    average = average/app_config.TRAIN_NUM
    # result[i]是type*words*score的三维组
    for i in range(len(result)):
        for j in range(len(result[i])):
            for k in range(len(result[i][j])):
                result[i][j][k] /= step
    return labels, sample, average, result


def create_plot(k, result):
    N = len(result[k])
    two_dimensional_matrix = np.empty((N, 10))
    for i in range(N):
        for j in range(10):
            two_dimensional_matrix[i, j] = result[k][i][j]
    # 创建图形
    fig = plt.figure(figsize=(10, 6))  # 设置图形的大小
    ax = fig.add_subplot(111, projection='3d')

    # 将二维数组中的点绘制为散点图
    for i in range(N):
        x = np.arange(10)  # x 坐标是 0 到 9
        y = np.repeat(i, 10)  # y 坐标是 i，重复 10 次
        z = two_dimensional_matrix[i, :]  # z 坐标是 two_dimensional_matrix 的一行数据

        ax.scatter(x, y, z)

    # 设置图形标题和标签
    ax.set_title('Three-Dimensional Scatter Plot of two_dimensional_matrix')
    ax.set_xlabel('type of words')
    ax.set_ylabel('kernal\'s position in words')
    ax.set_zlabel('score of kernel')

    # 显示图形
    plt.show()


def create_surf(k, result):
    N = len(result[k])
    two_dimensional_matrix = np.empty((N, 10))
    for i in range(N):
        for j in range(10):
            two_dimensional_matrix[i, j] = result[k][i][j]

    # 创建图形
    fig = plt.figure(figsize=(10, 6))  # 设置图形的大小
    ax = fig.add_subplot(111, projection='3d')

    # 创建 x, y 坐标网格
    x, y = np.meshgrid(np.arange(10), np.arange(N))

    # 将二维数组中的数据绘制为平滑曲面图
    ax.plot_surface(x, y, two_dimensional_matrix, cmap='viridis')

    # 设置图形标题和标签
    ax.set_title('Smooth Three-Dimensional Surface Plot')
    ax.set_xlabel('type of words')
    ax.set_ylabel('kernal\'s position in words')
    ax.set_zlabel('score of kernel')

    # 显示图形
    plt.show()


def create3D(average):
    # 创建 x, y 坐标
    x = np.arange(average.shape[0])  # 0 到 9
    y = np.arange(average.shape[1])  # 0 到 9

    # 创建网格
    X, Y = np.meshgrid(x, y)

    # 创建三维坐标系
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三维散点图
    ax.scatter(X, Y, average, c='r', marker='o')

    # 设置坐标轴标签
    ax.set_xlabel('True Labels')
    ax.set_ylabel('Predicted Labels')
    ax.set_zlabel('Score Predictions')

    plt.show()


def cos_dis(sample, average, labels):
    c = int(len(sample)/10)
    p = 0
    dis = np.zeros((sample.shape[0], average.shape[0]))
    for i in range(sample.shape[0]):
        # 对于每个平均行 j
        for j in range(average.shape[0]):
            dot_product = np.dot(sample[i], average[j])
            norm_sample = np.linalg.norm(sample[i])
            norm_average = np.linalg.norm(average[j])
            similarity = dot_product / (norm_sample * norm_average)
            dis[i][j] = similarity
    for j in range(10):
        suc = 0
        all = sample.shape[0]/10
        for i in range(c):
            max_value = max(dis[j*c+i])
            second_max_value = max(
                filter(lambda x: x != max_value, dis[j*c+i]), default=None)
            if np.argmax(dis[j*c+i]) == labels[j*c+i]:
                suc += 1
        print(f'第{j}类文余弦正确率:{suc/all}')
        p += suc/all
    print(f'余弦值总成功率:{p/10}')
    print()
    return dis


def average_dis(sample, labels):
    c = int(len(sample)/10)
    all = sample.shape[0]/10
    p = 0
    for j in range(10):
        suc = 0
        for i in range(c):
            if np.argmax(sample[i+j*c]) == labels[i+j*c]:
                suc += 1
        print(f'第{j}类平均值成功率:{suc/all}')
        p += suc/all
    print(f'平均值总成功率:{p/10}')
    print()
