# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2017-04-06'; 'last updated date: 2017-04-06'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function
from data_util1 import read_train_test_data
from feature_encoder import FeatureEncoder
from configure import Temp_Root_Path, Data_Type, Feature_Type, History_Length

__version__ = '1.0'

if __name__ == '__main__':
    data_type = Data_Type
    feature_type = Feature_Type

    # 读取训练数据和测试数据
    id_data_train, id_data_test, ood_data_train, ood_data_test, index_to_label_name = \
        read_train_test_data(data_type)

    # 造特征: 语法, 语义, 词汇等
    feature_encoder = FeatureEncoder(feature_type=feature_type,
                                     index_to_label_name=index_to_label_name,
                                     history_length=History_Length,
                                     verbose=0,
                                     )
    X_train, y_train = feature_encoder.fit_transform(data=(id_data_train, ood_data_train))
    # X_train = X_train.reshape(X_train.shape[0], -1)
    X_test, y_test = feature_encoder.transform(data=(id_data_test, ood_data_test))
    # X_test = X_test.reshape(X_test.shape[0], -1)
    # print(X_train.shape)
    # print(X_test.shape)
