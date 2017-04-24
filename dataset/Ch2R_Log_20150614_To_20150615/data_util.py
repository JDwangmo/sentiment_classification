# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2017-04-07'; 'last updated date: 2017-04-07'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function
import numpy as np
import pandas as pd
import sys
from os import path
from data_processing_util.cross_validation_util import get_k_fold_data
from data_processing_util.data_util import balance_data
from configure import Temp_Root_Path
from ood_dataset.stable_version.data_util import DataUtil
from configure import Data_Root_Path

__version__ = '1.0'


def read_train_test_data(data_type='id',history_length=0):
    """读取 训练和测试 数据
    
    Parameters
    ----------
    data_type:st

    Returns
    -------

    """
    id_data_train = pd.DataFrame()
    id_data_test = pd.DataFrame()
    ood_data_train = pd.DataFrame()
    ood_data_test = pd.DataFrame()
    id_index_to_label_name, ood_index_to_label_name = [], []
    # 读取 ID 句子
    if data_type in ['id', 'id_ood']:
        id_data_train, id_data_test = read_id_sentences(
            remove_redundant=False,
            train_test_split_rate=0.8,
            history_length=history_length,
            file_path=path.join(Data_Root_Path, 'ID标注-汇总-20170404.csv'),
            rand_seed=13,
        )
        # print(id_sentences.head())
        # print(id_sentences['Label'].value_counts())
        print(id_data_train['Label'].value_counts())
        print(id_data_test['Label'].value_counts())
        # quit()
        # print(id_data_train['Label'].value_counts() / id_sentences['Label'].value_counts())
        # print('id data_train: %d , id data_test: %d' % (len(id_data_train), len(id_data_test)))
        # quit()
        id_index_to_label_name = [u'陈述句',
                                  u'特指问',
                                  u'是非问',
                                  u'正反问',
                                  u'选择问',
                                  ]

    if data_type in ['ood', 'id_ood']:
        ood_data_train, ood_data_test, ood_index_to_label_name = read_ood_sentences()
    if data_type == 'id':
        data_train = id_data_train
        data_test = id_data_test
    elif data_type == 'ood':
        data_train = ood_data_train
        data_test = ood_data_test
    elif data_type == 'id_ood':
        data_train = pd.concat((ood_data_train, id_data_train))
        data_test = pd.concat((ood_data_test, id_data_test))
    else:
        raise NotImplementedError
    # print(data_train['Label'].value_counts())
    # print(data_test['Label'].value_counts())
    # print('data_train: %d , data_test: %d' % (len(data_train), len(data_test)))

    index_to_label_name = id_index_to_label_name + ood_index_to_label_name
    # print('类别数: %d' % len(index_to_label_name))
    return data_train, data_test, index_to_label_name


def read_id_sentences(file_path,
                      remove_redundant=False,
                      train_test_split_rate=0.8,
                      temp_root_path=Temp_Root_Path,
                      history_length=0,
                      rand_seed=5
                      ):
    """
    - 从文件中获取 ID 样例 
        - 每个样例格式为: [句子ID, ]
    - 检查数据 是否漏标, 
    - 统计异常对话

    Parameters
    ----------
    file_path: str
        文件路径
    remove_redundant: bool
        去重
    temp_root_path: str
        缓存文件夹
    train_test_split_rate: float
        训练和测试集合的切分比率
    history_length: int
        历史长度
            - history_length ==0 时, 将保存当前句的所有历史句
    rand_seed : int
        随机种子

    Returns 
    -------
    data: pd.DataFrame()
        ID sentences
    """
    log_data = pd.read_csv(
        file_path,
        sep='\t', doublequote=True, encoding='utf-8')

    print(log_data.shape)
    # print(log_data.head())
    # 过滤掉 空行
    # print(log_data.loc[log_data['SessionID'].isnull()])
    log_data = log_data.loc[log_data['SessionID'].notnull()]
    sys.stderr.write('总共有: %d 段对话\n' % len(set(log_data['SessionID'])))
    # print('\n'.join(set(log_data['SessionID'])))
    # quit()
    num_of_exception_dialogues = 0
    num_of_sentences = 0
    num_of_ood_sentences = 0
    num_of_id_sentences = 0
    num_of_ch2r_sentences = 0
    # 出现过的 话语+标签
    exited_sentences = set()
    rand = np.random.RandomState(rand_seed)
    data_train, data_test = [], []
    num_of_diaglogues_train, num_of_diaglogues_test = 0, 0

    for session_id, group in log_data.fillna('').groupby(by=['SessionID']):
        # region 1 检测异常对话,并去除
        if group.iloc[0][u'语义信息-备注'].__contains__(u'不要了'):
            # 检测 异常对话, 并直接跳过
            num_of_exception_dialogues += 1
            # print(session_id)
            continue
        # endregion

        rand_number = rand.rand()
        if rand_number > train_test_split_rate:
            datas = data_test
            # print(rand_number)
            num_of_diaglogues_test += 1
        else:
            datas = data_train
            num_of_diaglogues_train += 1
        history_user_sentences = []
        for sentence_index, (session_id,
                             time, name, record, sentence_mode,
                             sentence_mode_remark, semantic_info,
                             semantic_info_remark, remark
                             ) in enumerate(group.values):
            num_of_sentences += 1
            # region 2.1 - 检测是否有漏标
            if len(sentence_mode.strip()) == 0 and name != u'Ch2R':
                # 检测 是否有漏标 句型模式
                sys.stderr.write(u'漏标-句型模式: %s, %s, %s\n' % (session_id, name, record))
            if not sentence_mode.__contains__(u'协处理') and len(semantic_info.strip()) == 0 and name != u'Ch2R':
                # 检测 是否有漏标 语义信息
                sys.stderr.write(u'漏标-语义信息: %s, %s, %s\n' % (session_id, name, record))
            # endregion
            # region 2.2 - 跳过 OOD 话语 和 Ch2R的话语
            if name == u'Ch2R':
                num_of_ch2r_sentences += 1
                history_user_sentences.append(sentence_id)
                continue
            if sentence_mode.__contains__(u'协处理'):
                num_of_ood_sentences += 1
                continue
            # endregion
            num_of_id_sentences += 1
            record = record.strip()
            label = sentence_mode.split('+')[0].strip()
            # 去重 - 出现过就跳过
            if record + label in exited_sentences:
                continue
            else:
                if remove_redundant:
                    exited_sentences.add(record + label)
            sentence_id = u'%s-%d' % (session_id, sentence_index)
            datas.append([
                # 句子 id
                sentence_id,
                # 标签
                label,
                # 句子
                record,
                # 语义信息
                semantic_info,
                # user 历史对话
                ','.join(history_user_sentences[-history_length:]),
            ])
            history_user_sentences.append(sentence_id)

    sys.stderr.write('其中异常对话: %d\n' % num_of_exception_dialogues)
    sys.stderr.write('去除异常对话后,有: %d\n' % (len(set(log_data['SessionID'])) - num_of_exception_dialogues))
    sys.stderr.write('其中\n\t全部 话语: %d\n' % num_of_sentences)
    sys.stderr.write('\tCh2R 话语: %d\n' % num_of_ch2r_sentences)
    sys.stderr.write('\tOOD 话语: %d\n' % num_of_ood_sentences)

    sys.stderr.write('\tID 话语: %d\n' % num_of_id_sentences)
    data_train = pd.DataFrame(
        data=data_train,
        columns=[u'Sentence_ID', u'Label', u'Sentence', u'Semantic_Info', u'History_User_Sentences']
    )
    data_test = pd.DataFrame(
        data=data_test,
        columns=[u'Sentence_ID', u'Label', u'Sentence', u'Semantic_Info', u'History_User_Sentences']
    )
    sys.stderr.write('\ttrain 对话段: %d\n' % num_of_diaglogues_train)
    sys.stderr.write('\ttest 对话段: %d\n' % num_of_diaglogues_test)
    if remove_redundant:
        sys.stderr.write('\t去重后 ID 话语 - train: %d\n' % (len(data_train)))
        sys.stderr.write('\t去重后 ID 话语 - test: %d\n' % (len(data_test)))

    # region 保存
    temp_file_path = path.join(temp_root_path,
                               'id_sentences_HistoryLength%d_%s_train_%dsentences.csv' % (
                                   history_length,
                                   '去重' if remove_redundant else '未去重',
                                   len(data_train))
                               )
    write_dataframe_to_csv_file(data_train,
                                temp_file_path,
                                name='ID Sentences train '
                                )
    temp_file_path = path.join(temp_root_path,
                               'id_sentences_HistoryLength%d_%s_test_%dsentences.csv' % (
                                   history_length,
                                   '去重' if remove_redundant else '未去重',
                                   len(data_test))
                               )
    write_dataframe_to_csv_file(data_test,
                                temp_file_path,
                                name='ID Sentences test '
                                )
    # endregion

    return data_train, data_test


def read_ood_sentences():
    data_util = DataUtil()

    # region +++++++++++++  1 : 参数设置 +++++++++++++

    print('=' * 30 + ' 参数设置 ' + '=' * 30)

    config = {
        'dataset_type': 'v2.3(Sa)',
        'label_version': 'v2.0',
        'verbose': 1,
    }
    train_data, test_data = data_util.load_train_test_data(config)
    label_to_index, index_to_label = data_util.get_label_index(version=config['label_version'])

    # 转为和 ID数据 一致
    train_data['Sentence'] = train_data['SENTENCE']
    train_data['Label'] = train_data['LABEL']
    train_data['Sentence_ID'] = ['ood-%d' % idx for idx in range(len(train_data))]
    train_data['Semantic_Info'] = ''
    train_data['History_User_Sentences'] = ''
    train_data = train_data[['Sentence_ID', 'Label', 'Sentence', 'Semantic_Info', 'History_User_Sentences']]

    test_data['Sentence'] = test_data['SENTENCE']
    test_data['Label'] = test_data['LABEL']
    test_data['Sentence_ID'] = ['ood-%d' % idx for idx in range(len(test_data))]
    test_data['Semantic_Info'] = ''
    test_data['History_User_Sentences'] = ''
    test_data = test_data[['Sentence_ID', 'Label', 'Sentence', 'Semantic_Info', 'History_User_Sentences']]

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
