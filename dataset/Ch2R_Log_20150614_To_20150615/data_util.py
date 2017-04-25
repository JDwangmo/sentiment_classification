# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2017-04-07'; 'last updated date: 2017-04-18'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function
import numpy as np
import sys
from data_processing_util.cross_validation_util import get_k_fold_data
from id_dataset.stable_version.ch2r_id_dataset_util import read_train_test_data as read_id_train_test_data
from data_processing_util.data_util import balance_data
from ood_dataset.stable_version.data_util import DataUtil

__version__ = '1.0'


def read_train_test_data(data_type='id', history_length=0):
    """读取 训练和测试 数据
    
    Parameters
    ----------
    data_type:st

    Returns
    -------

    """
    id_data_train, id_data_test = None, None
    ood_data_train, ood_data_test = None, None
    id_index_to_label_name, ood_index_to_label_name = [], []
    # 读取 ID 句子
    if data_type.__contains__('id'):
        id_data_train, id_data_test, id_index_to_label_name = \
            read_id_train_test_data(data_type='Data_20150614To20150615_V1.0')

    if data_type.__contains__('ood'):
        ood_data_train, ood_data_test, ood_index_to_label_name = read_ood_sentences(verbose=1)

    index_to_label_name = id_index_to_label_name + ood_index_to_label_name
    # print('类别数: %d' % len(index_to_label_name))
    return id_data_train, id_data_test, ood_data_train, ood_data_test, index_to_label_name


def read_ood_sentences(verbose=0):
    data_util = DataUtil()

    # region +++++++++++++  1 : 参数设置 +++++++++++++

    print('=' * 30 + ' 参数设置 ' + '=' * 30)

    config = {
        'dataset_type': 'v2.3(Sa)',
        'label_version': 'v2.0',
        'verbose': verbose,
    }
    train_data, test_data = data_util.load_train_test_data(config)
    label_to_index, index_to_label = data_util.get_label_index(version=config['label_version'])

    # 转为和 ID数据 一致
    train_data['Sentence'] = train_data['SENTENCE']
    train_data['Label'] = train_data['LABEL']
    train_data['Sentence_ID'] = ['ood-%d' % idx for idx in range(len(train_data))]
    train_data = train_data[['Sentence_ID', 'Label', 'Sentence']]

    test_data['Sentence'] = test_data['SENTENCE']
    test_data['Label'] = test_data['LABEL']
    test_data['Sentence_ID'] = ['ood-%d' % idx for idx in range(len(test_data))]
    test_data = test_data[['Sentence_ID', 'Label', 'Sentence']]
    # region 1 类别的分布情况
    if verbose > 1:
        print('类别的分布情况')
        print(train_data['Label'].value_counts())
        print(test_data['Label'].value_counts())
        # endregion
    return train_data, test_data, index_to_label


def split_train_test_data(data, train_test_split_rate=0.8):
    # 打乱数据
    data_index = np.random.RandomState(5).permutation(len(data))
    # print(data_index)
    # quit()
    num_of_train_data = int(len(data) * train_test_split_rate)
    data_train, data_test = data.iloc[data_index[:num_of_train_data]], data.iloc[data_index[num_of_train_data:]]
    return data_train, data_test


def get_k_fold_train_test_data(X_train, y_train, X_test, y_test, k=3, sampling='over_sampling'):
    """
    
    Parameters
    ----------
    X_train: array-like
        特征矩阵
    y_train: array-like
        一维矩阵
    X_test: array-like
        特征矩阵
    y_test: array-like
        一维矩阵
    k: int
        几折交叉验证
    sampling: str
        抽样方法
    Returns
    -------

    """

    # print(train_x)
    # print(train_y)
    data = []
    for flag, X_train, y_train, X_test, y_test in get_k_fold_data(k=k,
                                                                  train_data=(X_train, y_train),
                                                                  test_data=(X_test, y_test),
                                                                  shuffle_data=True
                                                                  ):
        if sampling == 'over_sampling':
            X_train, y_train = balance_data(X_train, y_train)
        elif sampling == 'none':
            pass
        else:
            raise NotImplementedError
        data.append(
            [flag, X_train, y_train, X_test, y_test]
        )
    return data


def write_dataframe_to_csv_file(data, file_path, name=''):
    """
    将 DataFrame 数据对象 写入文件中

    Parameters
    ----------
    data: pd.DataFrame()
        数据
    file_path: str
        导出文件名
    name: str
        名称

    Returns
    -------

    """
    sys.stderr.write('%s 结果保存到: %s \n' % (name, file_path))
    data.to_csv(
        file_path,
        index=None,
        sep='\t',
        encoding='utf8'
    )


def print_info(info):
    sys.stderr.write('%s\n' % info)


if __name__ == '__main__':
    id_data_train, id_data_test, ood_data_train, ood_data_test, index_to_label_name = \
        read_train_test_data(data_type='id')
