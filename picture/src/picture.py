'''
Description: 
Author: 唐健峰
Date: 2023-09-28 22:00:51
LastEditors: ${author}
LastEditTime: 2023-09-28 23:34:26
'''
from picture.get_img import get_img_array_list

import numpy as np
from PIL import Image

Miyazaki_Hayao_image_list = get_img_array_list("resources/Miyazaki_Hayao/千寻")

# 将图像转换为NumPy数组
image_array = np.array(Miyazaki_Hayao_image_list[0])

# 将蓝色通道设为零
image_array[:, :, 2] = 0  # 第三个通道（蓝色通道）设为零


# 将 NumPy 数组转换为图像对象
save_image = Image.fromarray(image_array)

save_image.save('out/test1.jpg')
