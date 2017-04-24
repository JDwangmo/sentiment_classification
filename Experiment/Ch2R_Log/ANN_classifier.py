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
from configure import Temp_Root_Path, Data_Type, Feature_Type, Verbose, History_Length
from data_processing_util.evaluate import get_evaluate_score
import numpy as np

np.random.seed(1337)

__version__ = '1.0'

data_type = Data_Type
feature_type = Feature_Type
data_train, data_test, index_to_label_name = read_train_test_data(data_type=data_type)
# 造语法特征和语义特征
feature_encoder = FeatureEncoder(feature_type=feature_type,
                                 index_to_label_name=index_to_label_name,
                                 history_length=History_Length,
                                 verbose=Verbose,
                                 )
x_train, y_train = feature_encoder.fit_transform(data_train)
x_test, y_test = feature_encoder.transform(data_test)
print(x_train.shape)
print(x_test.shape)


# quit()
# print(data_train.head())
def fit(x_train, y_train, x_test, y_test, num_filter):
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    from keras.utils import np_utils

    data_dim = 21
    timesteps = History_Length + 1
    num_classes = 5

    y_train_onehot = np_utils.to_categorical(y_train, num_classes)
    y_test_onehot = np_utils.to_categorical(y_test, num_classes)
    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(32, return_sequences=True,
                   input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
    # model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(200))  # return a single vector of dimension 32
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
              batch_size=32, nb_epoch=50,
              validation_data=(x_test, y_test_onehot),
              verbose=2,
              # callbacks=[early_stop]
              )

    # region 保存结果
    # y_predict = model.predict_classes(x_test, batch_size=32, verbose=1)
    train_loss, train_accuracy = model.evaluate(x_train, y_train_onehot, verbose=0)
    val_loss, val_accuracy = model.evaluate(x_test, y_test_onehot, verbose=0)
    # print(y_predict)
    return train_loss, train_accuracy, val_loss, val_accuracy


# data_test.loc[:, 'Label_Predict'] = [feature_encoder.index_to_label_name[item] for item in y_predict]
# data_test.loc[:, 'Is_Correct'] = data_test['Label_Predict'] == data_test['Label']
# write_dataframe_to_csv_file(data_test,
#                             Temp_Root_Path + '/LSTM_%s_%s_HistoryLength%d_test_result.csv' % (
#                                 data_type, feature_type, History_Length), 'LSTM test result')

# endregion



from data_processing_util.cross_validation_util import transform_cv_data, get_k_fold_data, get_val_score

# 1. 获取交叉验证的数据
cv_data = get_k_fold_data(
    k=3,
    train_data=(x_train, y_train),
    test_data=(x_test, y_test),
    include_train_data=True,
)

cv_data = [item + [None] for item in cv_data]
# 交叉验证
for num_filter in [10, 20, 30, 100, 200, 500]:
    print('=' * 40)
    print('num_filter is %d.' % num_filter)

    # K折
    print('K折交叉验证开始...')
    # counter = 0
    test_acc = []
    val_result = []
    train_acc = []
    conv_middle_output_dev = []
    conv_middle_output_val = []
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
        # 构建分类器对象
        # print(parameters)
        # estimator.print_model_descibe()

        # 拟合数据
        # dev_loss, dev_accuracy, val_loss, val_accuracy = 0,0,0,0
        dev_loss, dev_accuracy, val_loss, val_accuracy = fit(dev_X, dev_y, val_X, val_y, num_filter)
        # y_pred = estimator.batch_predict(val_X, transform_input=False)
        # val_result.append(y_pred)

        # if parameters.get('get_cnn_middle_layer_output', False):
        #     获取中间层输出
        # conv_middle_output_dev.append(estimator.get_layer_output(dev_X))
        # conv_middle_output_val.append(estimator.get_layer_output(val_X))

        print('dev:%f,%f' % (dev_loss, dev_accuracy))
        print('val:%f,%f' % (val_loss, val_accuracy))

        test_acc.append(val_accuracy)
        train_acc.append(dev_accuracy)
        # if not parameters.get('need_validation', 'True'):
        #     break
        # counter += 1
        quit()

    print('k折验证结果：%s' % test_acc)
    print('验证中训练数据结果：%s' % train_acc)
    print('验证中测试数据平均准确率：%f' % np.average(test_acc[int(exclude_first):]))
    print('测试结果汇总：%s' % (test_acc + [np.average(test_acc[int(exclude_first):])]))
    print('%s,%s' % (train_acc, test_acc))
    print('-' * 80)
