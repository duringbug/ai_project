from cloud.duringbug.dao.data import *
from cloud.duringbug.preprocessing.index import *
from cloud.duringbug.preprocessing.read import train_file_divide
from cloud.duringbug.train.multilayer_perceptron_train import *

import torch


def before_multilayer_perceptron():
    train_file_divide()
    dbinit()
    tf_idf_Bow()
    entropy_BoW()
    score_init()


def multilayer_perceptron():
    words_train, labels_train = train_txt_to_multilayer_perceptron_sample(
        "resources/exp1_data/my_train_data.txt")
    words_test, labels_test = train_txt_to_multilayer_perceptron_sample(
        "resources/exp1_data/my_verification_data.txt")
    pridict_sample = trans_test_txt_to_multilayer_perceptron_sample(
        "resources/exp1_data/my_test.txt")
    pridict_sample = torch.tensor(pridict_sample, dtype=torch.float32)
    model = tain_multilayer_perceptron(
        words_train, labels_train, words_test, labels_test)
    with torch.no_grad():
        predictions = model(pridict_sample)
        _, predicted_labels = torch.max(predictions, dim=1)  # 获取预测类别
    with open('out/results.txt', 'w') as file:
        file.write("id, pred\n")
        for row_index, label in enumerate(predicted_labels):
            file.write("{} {}\n".format(row_index, label))


def after_multilayer_perceptron():
    try:
        os.remove("out/results.txt")
        print(f"文件 'out/results.txt' 删除成功")
    except OSError as e:
        print(f"Error deleting file: {e}")
    try:
        os.remove("BoW.db")
        print(f"文件 'BoW.db' 删除成功")
    except OSError as e:
        print(f"Error deleting file: {e}")
