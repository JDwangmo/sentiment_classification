# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2017-04-06'; 'last updated date: 2017-04-06'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function
from configure import Data_File_Path, Temp_Root_Path
from feature_encoder import FeatureEncoder
from sklearn.ensemble.forest import RandomForestClassifier
from data_util import read_id_sentences, get_k_fold_train_test_data, split_train_test_data, write_dataframe_to_csv_file
from data_processing_util.evaluate import get_evaluate_score
import numpy as np

__version__ = '1.0'

if __name__ == '__main__':
    # 读取 ID 句子
    id_sentences = read_id_sentences(
        remove_redundant=True,
        file_path=Data_File_Path
    )
    # 造语法特征和语义特征

    # print(id_sentences.head())
    data_train, data_test = split_train_test_data(id_sentences, train_test_split_rate=0.8)
    # print(data_train.head())
    print(id_sentences['Label'].value_counts())
    # print(data_train['Label'].value_counts())
    # print(data_test['Label'].value_counts())
    print(data_train['Label'].value_counts() / id_sentences['Label'].value_counts())
    # quit()
    feature_encoder = FeatureEncoder()
    X_train, y_train = feature_encoder.fit_transform(data_train)
    X_test, y_test = feature_encoder.transform(data_test)
    # 获取 交叉数据
    # data = get_k_fold_train_test_data(X_train, y_train, X_test, y_test, k=3, sampling='none')
    # print(train_x)
    # print(train_y)

    # rf_model = RandomForestClassifier(n_estimators=500, random_state=0)
    #
    # # get_k_fold_data(k=2, train_data=)
    # accuracy_list = []
    # for flag, X_train, y_train, X_test, y_test in data:
    #     print(len(X_train), len(X_test))
    #     rf_model.fit_transform(X_train, y_train)
    #     y_predict = rf_model.predict(X_test)
    #     print(y_predict)
    #     print(y_test)
    #     accuracy = get_evaluate_score(y_test, y_predict)
    #     if flag == 0:
    #         data_test.loc[:, 'Label_Predict'] = [feature_encoder.index_to_label_name[item] for item in y_predict]
    #         data_test.loc[:, 'Is_Correct'] = data_test['Label_Predict'] == data_test['Label']
    #         write_dataframe_to_csv_file(data_test, Temp_Root_Path + '/test_result.csv')
    #     else:
    #         accuracy_list.append(accuracy)
    #     quit()
    # print(np.mean(accuracy_list))
