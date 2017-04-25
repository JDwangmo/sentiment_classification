# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2017-04-11'; 'last updated date: 2017-04-11'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function
from dataset.Ch2R_Log_20150614_To_20150615.data_util import get_k_fold_train_test_data, \
    write_dataframe_to_csv_file, read_train_test_data, print_info
from dataset.Ch2R_Log_20150614_To_20150615.feature_encoder import FeatureEncoder
from configure import Temp_Root_Path, Data_Type, Feature_Type, Verbose, History_Length, Nb_Epoch, Data_Root_Path
from data_processing_util.evaluate import get_evaluate_score
from data_processing_util.cross_validation_util import get_k_fold_data
import numpy as np
import pandas as pd
import pickle
import sys
from os import path
import time

__version__ = '1.0'


def fit(x_train, y_train, x_test, y_test, num_filter1, num_filter2):
    np.random.seed(1337)
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    from keras.utils import np_utils

    data_dim = 21
    timesteps = History_Length + 1
    num_classes = 5
    nb_epoch = Nb_Epoch

    y_train_onehot = np_utils.to_categorical(y_train, num_classes)
    y_test_onehot = np_utils.to_categorical(y_test, num_classes)
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(num_filter1, return_sequences=True,
                   input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    # model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(num_filter2))  # return a single vector of dimension 32
    # model.add(Dense(200, activation='tanh'))
    model.add(Dense(num_classes, activation='softmax'))
    if Verbose > 0:
        model.summary()
    from keras.optimizers import SGD, Adadelta
    from keras.callbacks import EarlyStopping

    optimizers = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    # optimizers = Adadelta(lr=1e-6, rho=0.95, epsilon=1e-6)
    # optimizers = 'adadelta'
    # model.compile(loss='categorical_crossentropy', optimizer=optimizers, metrics=['accuracy'])
    # early_stop = EarlyStopping(patience=30, verbose=2)

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(x_train, y_train_onehot,
              batch_size=32, nb_epoch=nb_epoch,
              validation_data=(x_test, y_test_onehot),
              verbose=Verbose,
              # callbacks=[early_stop]
              )

    # region 保存结果
    # y_predict = model.predict_classes(x_test, batch_size=32, verbose=1)
    train_loss, train_accuracy = model.evaluate(x_train, y_train_onehot, verbose=0)
    val_loss, val_accuracy = model.evaluate(x_test, y_test_onehot, verbose=0)
    y_val_pred = model.predict_classes(x_test, batch_size=32, verbose=0)
    # print(y_predict)
    return train_loss, train_accuracy, val_loss, val_accuracy, y_val_pred


def run_lstm_classifier():
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
        fout = open(temp_path, 'wb')
        pickle.dump(
            [X_train, y_train, X_test, y_test],
            fout
        )
        pickle.dump(
            [id_data_train, id_data_test, ood_data_train, ood_data_test, index_to_label_name],
            fout
        )
        sys.stderr.write('将数据缓存到文件中: %s\n' % temp_path)
    else:
        sys.stderr.write('从缓存文件中加载数据: %s\n' % data_path)
        fin = open(data_path, 'rb')
        X_train, y_train, X_test, y_test = pickle.load(
            fin
        )
        # id_data_train, id_data_test, ood_data_train, ood_data_test, index_to_label_name = pickle.load(
        #     open(data_path, 'rb')
        # )

    print(X_train.shape)
    print(X_test.shape)
    # endregion

    # quit()
    # print(data_train.head())

    # 1. 获取交叉验证的数据
    cv_data = get_k_fold_data(
        k=3,
        train_data=(X_train, y_train),
        test_data=(X_test, y_test),
        include_train_data=True,
    )

    cv_data = [item + [None] for item in cv_data]
    val_result_dict = {}
    # 交叉验证
    # for num_filter1 in [10, 20, 30, 50, 60, 100, 150, 200, 500]:
    for num_filter1 in [500]:
        # for num_filter2 in [10, 20, 30, 60, 100, 150, 200, 300]:
        for num_filter2 in [ 200, 300]:
            print('=' * 40)
            start = time.time()
            print('num_filter is %d,%d.' % (num_filter1, num_filter2))
            val_result_dict['%d-%d' % (num_filter1, num_filter2)] = []
            # K折
            print('K折交叉验证开始...')
            # counter = 0
            test_acc = []
            val_result = []
            train_acc = []
            exclude_first = False
            for flag, dev_X, dev_y, val_X, val_y, feature_encoder in cv_data:
                # print(len(dev_X))
                print('-' * 80)
                if flag == 0:
                    # 第一个数据是训练，之后是交叉验证
                    print('训练:')
                    # 因为第一份是训练排除掉 (后面要对交叉验证结果进行平均，将训练集的结果排除掉)
                    exclude_first = True
                else:
                    print('第%d个验证' % flag)
                    # continue
                # 构建分类器对象
                # print(parameters)
                # estimator.print_model_descibe()

                # 拟合数据
                # dev_loss, dev_accuracy, val_loss, val_accuracy = 0,0,0,0
                dev_loss, dev_accuracy, val_loss, val_accuracy, y_val_pred = fit(dev_X, dev_y, val_X, val_y,
                                                                                 num_filter1, num_filter2)
                # y_pred = batch_predict(val_X, transform_input=False)
                # val_result.append(y_pred)
                val_result_dict['%d-%d' % (num_filter1, num_filter2)] += [y_val_pred]

                print('dev:%f,%f' % (dev_loss, dev_accuracy))
                print('val:%f,%f' % (val_loss, val_accuracy))

                test_acc.append(val_accuracy)
                train_acc.append(dev_accuracy)
                # if not parameters.get('need_validation', 'True'):
                #     break
                # counter += 1
                # quit()
            end = time.time()
            print('用时：%f' % (end - start))
            print('k折验证结果：%s' % test_acc)
            print('验证中训练数据结果：%s' % train_acc)
            print('验证中测试数据平均准确率：%f' % np.average(test_acc[int(exclude_first):]))
            print('测试结果汇总：%s' % (test_acc + [np.average(test_acc[int(exclude_first):])]))
            print('%s,%s' % (train_acc, test_acc))
            print('-' * 80)

    # region 保存结果
    n_filter = '10-150'
    y_predict = val_result_dict[n_filter][0]
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
    data_test.loc[:, 'Label_Predict'] = [in_feature_encoder.index_to_label_name[item] for item in y_predict]
    data_test.loc[:, 'Is_Correct'] = data_test['Label_Predict'] == data_test['Label']
    write_dataframe_to_csv_file(
        data_test,
        Temp_Root_Path + '/LSTM%s_%s_%s_HistoryLength%d_test_result.csv' % (
            n_filter, data_type, feature_type, History_Length),
        'rf_50 test result')
    # endregion


if __name__ == '__main__':
    run_lstm_classifier()
