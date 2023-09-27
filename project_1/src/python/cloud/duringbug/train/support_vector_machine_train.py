from cloud.duringbug.dao.index import get_words_zero_dict
from cloud.duringbug.preprocessing.read import readPreprocessing
from cloud.duringbug.preprocessing.index import split_text
from cloud.duringbug.dao.data import get_entropy_inBow
from tqdm import tqdm
import json
import numpy as np


def train_txt_to_support_vector_machine_sample(path):
    data = readPreprocessing(path)
    json_objects = data.strip().split('\n')
    labels = np.zeros((1, len(json_objects)))
    words_zero_dict = get_words_zero_dict()
    entropy = get_entropy_inBow()
    words = []
    for i, json_str in enumerate(tqdm(json_objects, total=len(json_objects), desc=f'遍历{path},数据转成词袋(0,1)向量')):
        word = words_zero_dict.copy()
        data = json.loads(json_str)
        label = data["label"]
        raw_text = data["raw"]
        labels[0][i] = label
        punctuation_to_split = r'[| -,&!".:?();\n$\'#\*-+]+(?!\s)|\s+'
        target_words = split_text(raw_text, punctuation_to_split)
        for target_word in target_words:
            if target_word in word:
                word[target_word] += 1
        words.append(word)
    return np.array([list(word.values()) for word in words])/entropy, labels


def trans_test_txt_to_support_vector_machine_sample(path):
    words_zero_dict = get_words_zero_dict()
    entropy = get_entropy_inBow()
    words = []
    # 定义一个空字典来存储文本数据
    text_dict = {}
    with open(path, 'r') as file:
        for line in file:
            word = words_zero_dict.copy()
            parts = line.strip().split(', ', 1)
            text = parts[1]
            punctuation_to_split = r'[| -,&!".:?();\n$\'#\*-+]+(?!\s)|\s+'
            target_words = split_text(text, punctuation_to_split)
            for target_word in target_words:
                if target_word in word:
                    word[target_word] += 1
            words.append(word)
    return np.array([list(word.values()) for word in words])/entropy
