# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-01'
    Email:   '383287471@qq.com'
    Describe: 单卷积层的 CNN（rand-w2v）,
                模型编号： CNN_A10,
                具体参考 /dataset/模型汇总.xlsx,
                结果见：/MR/result/MR_CNN-A00_cnn-w2v_cv.xls
        1. 读取数据 MR数据集
        2. 交叉验证
"""

# region -------------- 0、参数设置 -------------
# option = 'cv'
option = 'output_cv_result'

# endregion -------------- 0、参数设置 ---------------


# region 1、加载数据集
from dataset.data_util import DataUtil
import numpy as np

data_version = 'Subj'
data_util = DataUtil()
train_data = data_util.get_train_test_data(version=data_version)

train_x = train_data['TEXT'].as_matrix()
train_y = train_data['LABEL'].as_matrix()

# endregion

# region 2、交叉验证
if option == 'cv':
    from deep_learning.cnn.wordEmbedding_cnn.example.one_conv_layer_wordEmbedding_cnn import WordEmbeddingCNNWithOneConv
    # 句子最长长度为：120
    input_length = 128
    word_embedding_dim = 300
    WordEmbeddingCNNWithOneConv.cross_validation(
        train_data=(train_x, train_y),
        test_data=(train_x, train_y),
        include_train_data=False,
        need_validation=True,
        vocabulary_including_test_set=True,
        # 设置权重随机化
        rand_weight=True,
        # 设置权重可训练
        embedding_weight_trainable=True,
        cv=10,
        num_labels=2,
        need_segmented=False,
        batch_size=50,
        lr=8e-3,
        input_length=input_length,
        num_filter_list=[100],
        # num_filter_list=[10,30,50, 80, 100, 110, 150, 200, 300,500,1000],
        verbose=1,
    )
# endregion


# region -------------- 3、提取出验证结果的数据 -------------
if option == 'output_cv_result':
    from dataset.data_util import output_validation_result

    output_validation_result(
        path='/home/jdwang/PycharmProjects/sentiment_classification/Subj/result/Subj_CNN-A10_cnn-rand_cv.txt',
        version='CNN-A00',
        step=4
    )
# endregion -------------- 3、提取出验证结果的数据 ---------------
