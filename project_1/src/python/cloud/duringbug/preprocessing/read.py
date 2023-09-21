'''
Description: 
Author: 唐健峰
Date: 2023-09-14 10:21:01
LastEditors: ${author}
LastEditTime: 2023-09-19 18:08:26
'''
import os
import random
from cloud.duringbug.conf.config import AppConfig

app_config = AppConfig()
app_config.load_config_from_file('resources/config/config.json')


def readPreprocessing(path):
    with open(path, 'r') as file:
        # 读取文件内容
        file_contents = file.read()
    return file_contents


def train_file_divide():
    try:
        # 删除文件
        os.remove('resources/exp1_data/my_train_data.txt')
        os.remove('resources/exp1_data/my_verification_data.txt')
    except FileNotFoundError:
        print(f'文件my_train_data不存在。')
    except Exception as e:
        print(f'删除文件my_train_data时发生错误:{e}')
    with open('resources/exp1_data/my_train_data.txt', 'w') as file:
        file.write('')
    with open('resources/exp1_data/my_verification_data.txt', 'w') as file:
        file.write('')

    with open('resources/exp1_data/train_data.txt', 'r') as file:
        content = file.read()

    # 将字符串分割为多个JSON对象
    json_objects = content.strip().split('\n')

    all_num = len(json_objects)
    # 生成0到799的整数集合
    all_numbers = set(range(800))

    # 随机选择不重复的整数：注意index.py的63行也要改
    random_numbers = set(random.sample(all_numbers, app_config.TRAIN_NUM))

    # 计算不包含在随机选择中的整数集合
    remaining_numbers = all_numbers - random_numbers

    # 转换为列表形式
    remaining_numbers_list = list(remaining_numbers)

    with open('resources/exp1_data/my_train_data.txt', 'w') as file1:
        with open('resources/exp1_data/my_verification_data.txt', 'w') as file2:
            for i in range(0, 10):
                for random_number in random_numbers:
                    file1.write(json_objects[i*800+random_number]+'\n')
                for remaining_number in remaining_numbers_list:
                    file2.write(json_objects[i*800+remaining_number]+'\n')
    print(f"训练集{app_config.TRAIN_NUM*10}条与测试集{(800-app_config.TRAIN_NUM)*10}条划分成功")
    return 0
