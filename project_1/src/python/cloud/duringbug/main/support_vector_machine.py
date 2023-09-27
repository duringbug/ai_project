'''
Description: 
Author: 唐健峰
Date: 2023-09-26 14:09:17
LastEditors: ${author}
LastEditTime: 2023-09-27 08:53:33
'''
from cloud.duringbug.dao.data import *
from cloud.duringbug.preprocessing.index import *
from cloud.duringbug.preprocessing.read import train_file_divide
from cloud.duringbug.train.support_vector_machine_train import train_txt_to_support_vector_machine_sample
from cloud.duringbug.train.support_vector_machine_train import trans_test_txt_to_support_vector_machine_sample

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def before_support_vector_machine():
    train_file_divide()
    dbinit()
    tf_idf_Bow()
    entropy_BoW()
    score_init()


def support_vector_machine():
    words_train, labels_train = train_txt_to_support_vector_machine_sample(
        "resources/exp1_data/my_train_data.txt")
    words_test, labels_test = train_txt_to_support_vector_machine_sample(
        "resources/exp1_data/my_verification_data.txt")
    pridict_sample = trans_test_txt_to_support_vector_machine_sample(
        "resources/exp1_data/my_test.txt")
    # 创建SVM分类器
    svm_classifier = SVC(kernel='linear', C=1)
    print("使用训练集训练SVM模型中...")
    # 使用训练集训练SVM模型
    svm_classifier.fit(words_train, labels_train.reshape(-1))
    print("使用模型对测试集进行预测...")
    # 使用模型对测试集进行预测
    predictions = svm_classifier.predict(words_test)

    # 计算准确度
    accuracy = accuracy_score(labels_test.reshape(-1), predictions)
    print("模型的准确度:", accuracy)

    # 计算预测集
    test_predictions = svm_classifier.predict(pridict_sample)
    with open('out/results.txt', 'w') as file:
        file.write("id, pred\n")
        for row_index, predict_type in enumerate(test_predictions):
            file.write("{} {}\n".format(row_index, int(predict_type)))


def after_support_vector_machine():
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
