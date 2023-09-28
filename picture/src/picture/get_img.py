'''
Description: 
Author: 唐健峰
Date: 2023-09-28 22:46:27
LastEditors: ${author}
LastEditTime: 2023-09-28 22:48:07
'''
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")


def get_img_array_list(path):
    folder_path = path
    image_list = []
    num_files = sum(1 for filename in os.listdir(folder_path)
                    if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')))

    # 遍历文件夹下的所有文件
    with tqdm(total=num_files, desc=f'遍历{path}下的图片') as pbar:
        for filename in os.listdir(folder_path):
            # 检查文件扩展名，确保只处理图像文件（例如，.jpg、.png 等）
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                # 构建完整的文件路径
                file_path = os.path.join(folder_path, filename)

                # 打开图像文件并将其添加到列表中
                image = Image.open(file_path)
                image_list.append(image)
                # 更新 tqdm 进度条
                pbar.update(1)
    return image_list
