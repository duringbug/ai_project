from cloud.duringbug.dao.data import *
from cloud.duringbug.preprocessing.index import *
from cloud.duringbug.preprocessing.read import train_file_divide


def before_multilayer_perceptron():
    train_file_divide()
    dbinit()
    tf_idf_Bow()
    entropy_BoW()
    score_init()
