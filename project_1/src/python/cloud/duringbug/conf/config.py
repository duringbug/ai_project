'''
Description: 
Author: 唐健峰
Date: 2023-09-19 17:39:36
LastEditors: ${author}
LastEditTime: 2023-09-19 17:47:19
'''

import json


class AppConfig:
    # 默认配置参数
    TRAIN_NUM = 400
    SECRET_KEY = '123456'

    # 初始化方法
    def __init__(self, config_file=None):
        # 如果提供了配置文件，从配置文件加载配置
        if config_file:
            self.load_config_from_file(config_file)

    # 从配置文件加载配置
    def load_config_from_file(self, config_file):
        with open(config_file, 'r') as file:
            config_data = json.load(file)
            for key, value in config_data.items():
                setattr(self, key, value)
