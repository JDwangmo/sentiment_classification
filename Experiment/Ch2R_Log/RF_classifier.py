# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2017-04-08'; 'last updated date: 2017-04-08'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function
from dataset.Ch2R_Log_20150614_To_20150615.data_util import get_k_fold_train_test_data, \
    write_dataframe_to_csv_file, read_train_test_data, print_info
from dataset.Ch2R_Log_20150614_To_20150615.feature_encoder import FeatureEncoder
from configure import Temp_Root_Path, Data_Type, Feature_Type, Verbose, History_Length
from traditional_classify.bow_rf.bow_rf_model import BowRandomForest
import sys

__version__ = '1.0'


def run_rf_classifier():
    data_type = Data_Type
    feature_type = Feature_Type
    data_train, data_test, index_to_label_name = read_train_test_data(data_type=data_type,
                                                                      history_length=History_Length,
                                                                      )
    # 造语法特征和语义特征
    feature_encoder = FeatureEncoder(feature_type=feature_type,
                                     index_to_label_name=index_to_label_name,
                                     history_length=History_Length,
                                     verbose=Verbose,
                                     )
    X_train, y_train = feature_encoder.fit_transform(data_train)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test, y_test = feature_encoder.transform(data_test)
    X_test = X_test.reshape(X_test.shape[0], -1)
    print(X_train.shape)
    print(X_test.shape)
    # quit()
    val_result_dict = BowRandomForest.cross_validation(
        train_data=(X_train, y_train),
        test_data=(X_test, y_test),
        shuffle_data=True,
        n_estimators_list=[10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000],
        # n_estimators_list=[50],
        need_transform_input=False,
        # feature_type=feature_type,
        # word2vec_to_solve_oov=False,
        # word2vec_model_file_path=None,
        # need_segmented=True,
        verbose=1,
        cv=3,
        need_validation=True,
        include_train_data=True,
        get_predict_result=True
    )
    # region 保存结果
    y_predict = val_result_dict['50'][0]
    data_test.loc[:, 'Label_Predict'] = [feature_encoder.index_to_label_name[item] for item in y_predict]
    data_test.loc[:, 'Is_Correct'] = data_test['Label_Predict'] == data_test['Label']
    write_dataframe_to_csv_file(
        data_test,
        Temp_Root_Path + '/RF_%s_%s_HistoryLength%d_test_result.csv' % (data_type, feature_type, History_Length),
        'rf_50 test result')
    # endregion


if __name__ == '__main__':
    run_rf_classifier()
