# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-09-11'
    Email:   '383287471@qq.com'
    Describe: 单卷积层的 CNN（rand-w2v）,
                模型编号： RF_A10,
                具体参考 /dataset/模型汇总.xlsx,
                结果见：/MR/result/RF-A10_rf-bow_cv.xls
        1. 读取数据 MR数据集
        2. 交叉验证
"""

# region 1、加载数据集
from dataset.data_util import DataUtil

data_util = DataUtil()
train_data = data_util.get_train_test_data(version='MR')

train_x = train_data['TEXT'].as_matrix()
train_y = train_data['LABEL'].as_matrix()
# endregion


from traditional_classify.bow_rf.bow_rf_model import BowRandomForest

BowRandomForest.cross_validation(
    train_data=(train_x, train_y),
    test_data=(train_x, train_x),
    shuffle_data=False,
    include_train_data=False,
    need_segmented=False,
    need_validation=True,
    cv=10,
    n_estimators_list = [10,20,30,40,50,60,70,80,90,100,200,300,400,500,1000,2000,3000,4000,5000],
    # n_estimators_list = [640,470,490],
    # n_estimators_list=[290],
    # n_estimators_list = range(10,1010,10),
    # n_estimators_list = [330],
    verbose=0,
    # feature_type = 'word',
    word2vec_to_solve_oov=False,
    word2vec_model_file_path=data_util.transform_word2vec_model_name('%300d_google_news'),
)
