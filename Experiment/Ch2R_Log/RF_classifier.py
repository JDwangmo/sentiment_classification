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
from configure import Temp_Root_Path, Data_Type, Feature_Type, Verbose, History_Length, Data_Root_Path
from traditional_classify.bow_rf.bow_rf_model import BowRandomForest
import pandas as pd
import pickle
import sys
from os import path

__version__ = '1.0'


def run_rf_classifier():
    data_type = Data_Type
    feature_type = Feature_Type
    # region 读取训练数据和测试数据
    data_path = Data_Root_Path + '/%s_%s_HistoryLength%d_Train&TestData.csv' % (data_type, feature_type, History_Length)
    if not path.exists(data_path):
        id_data_train, id_data_test, ood_data_train, ood_data_test, index_to_label_name = \
            read_train_test_data(data_type)

        # 造特征: 语法, 语义, 词汇等
        in_feature_encoder = FeatureEncoder(feature_type=feature_type,
                                            index_to_label_name=index_to_label_name,
                                            history_length=History_Length,
                                            verbose=0,
                                            temp_root_path=Temp_Root_Path
                                            )
        X_train, y_train = in_feature_encoder.fit_transform(data=(id_data_train, ood_data_train))
        X_test, y_test = in_feature_encoder.transform(data=(id_data_test, ood_data_test))
        temp_path = Temp_Root_Path + '/%s_%s_HistoryLength%d_Train&TestData.csv' % (
            data_type, feature_type, History_Length)
        pickle.dump(
            [X_train, y_train, X_test, y_test, id_data_train, id_data_test, ood_data_train, ood_data_test,
             index_to_label_name],
            open(temp_path, 'wb')
        )
        sys.stderr.write('将数据缓存到文件中: %s\n' % temp_path)
    else:
        sys.stderr.write('从缓存文件中加载数据: %s\n' % data_path)
        X_train, y_train, X_test, y_test, id_data_train, id_data_test, \
          ood_data_train, ood_data_test, index_to_label_name = pickle.load(
            open(data_path, 'rb')
        )

    print(X_train.shape)
    print(X_test.shape)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    # endregion

    # quit()
    val_result_dict = BowRandomForest.cross_validation(
        train_data=(X_train, y_train),
        test_data=(X_test, y_test),
        shuffle_data=True,
        # n_estimators_list=[10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000],
        n_estimators_list=[300],
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
    n_tree = '300'
    y_predict = val_result_dict[n_tree][0]
    sentences = []
    y_true = []
    if data_type.__contains__('id'):
        sentences += [utt.sentence for utt in id_data_test.iter_utterances(data_type='id')]
        y_true += [utt.sentence_mode for utt in id_data_test.iter_utterances(data_type='id')]
    if data_type.__contains__('ood'):
        sentences += ood_data_test['Sentence'].tolist()
        y_true += ood_data_test['Label'].tolist()

    data_test = pd.DataFrame(
        data={'Sentence': sentences,
              'Label': y_true,
              }
    )
    print(data_test.shape)
    data_test.loc[:, 'Label_Predict'] = [index_to_label_name[item] for item in y_predict]
    data_test.loc[:, 'Is_Correct'] = data_test['Label_Predict'] == data_test['Label']
    write_dataframe_to_csv_file(
        data_test,
        Temp_Root_Path + '/RF%s_%s_%s_HistoryLength%d_test_result.csv' % (
            n_tree, data_type, feature_type, History_Length),
        'rf_50 test result')
    # endregion


if __name__ == '__main__':
    run_rf_classifier()
