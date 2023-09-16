from cloud.duringbug.dao.data import *
from cloud.duringbug.preprocessing.index import main
from cloud.duringbug.train.train_txt_processing import *
from cloud.duringbug.preprocessing.read import train_file_divide


def index():
    train_file_divide()
    init()
    main()
    entropy_BoW()
    score_init()
    test_txt_to_matrix()
    test_txt_to_matrix_without_b()
