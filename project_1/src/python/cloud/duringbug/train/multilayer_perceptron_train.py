'''
Description: 
Author: 唐健峰
Date: 2023-09-26 13:52:08
LastEditors: ${author}
LastEditTime: 2023-09-28 12:57:54
'''
from cloud.duringbug.dao.index import get_words_zero_dict
from cloud.duringbug.preprocessing.read import readPreprocessing
from cloud.duringbug.preprocessing.index import split_text
from cloud.duringbug.dao.data import get_entropy_inBow
from tqdm import tqdm
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def train_txt_to_multilayer_perceptron_sample(path):
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


def trans_test_txt_to_multilayer_perceptron_sample(path):
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


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# words_train(4000*60000),labels_train(4000*1)
# words_test(4000*60000), labels_test(4000*1)


def tain_multilayer_perceptron(words_train, labels_train, words_test, labels_test):
    # 转换数据为PyTorch张量
    words_train = torch.tensor(words_train, dtype=torch.float32)
    labels_train = torch.tensor(
        labels_train, dtype=torch.long).squeeze()  # 使用 long 类型标签
    words_test = torch.tensor(words_test, dtype=torch.float32)
    labels_test = torch.tensor(
        labels_test, dtype=torch.long).squeeze()  # 使用 long 类型标签

    # 创建MLP模型
    input_dim = words_train.shape[1]
    hidden_dim = 64
    output_dim = 10  # 假设有10个类别
    num_epochs = 100
    learning_rate = 0.001
    model = MLP(input_dim, hidden_dim, output_dim)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(words_train)
        loss = criterion(outputs, labels_train)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 训练集上的准确率
    with torch.no_grad():
        train_predictions = model(words_train)
        _, train_predicted_labels = torch.max(
            train_predictions, dim=1)  # 获取训练集预测类别
        train_correct_predictions = (
            train_predicted_labels == labels_train).sum().item()
        train_total_samples = labels_train.size(0)
        train_accuracy = train_correct_predictions / train_total_samples
        print(f'Accuracy on Training Data: {train_accuracy:.4f}')
    # 测试模型
    with torch.no_grad():
        predictions = model(words_test)
        _, predicted_labels = torch.max(predictions, dim=1)  # 获取预测类别
        correct_predictions = (predicted_labels == labels_test).sum().item()
        total_samples = labels_test.size(0)
        accuracy = correct_predictions / total_samples
        print(f'Accuracy on Test Data: {accuracy:.4f}')
    return model
