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

__version__ = '1.0'


def read_id_sentences(file_path, remove_redundant=False, temp_root_path=Temp_Root_Path):
    """
    - 从文件中获取 ID 样例 
        - 每个样例格式为: [句子ID, ]
    - 检查数据 是否漏标, 
    - 统计异常对话

    Parameters
    ----------
    temp_root_path: str
        缓存文件夹
    remove_redundant: bool
        去重
    file_path: str
        文件路径

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
    datas = []
    for session_id, group in log_data.fillna('').groupby(by=['SessionID']):
        # region 1 检测异常对话,并去除
        if group.iloc[0][u'语义信息-备注'].__contains__(u'不要了'):
            # 检测 异常对话, 并直接跳过
            num_of_exception_dialogues += 1
            # print(session_id)
            continue
        # endregion
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
            datas.append([
                # 句子 id
                u'%s-%d' % (session_id, sentence_index),
                # 标签
                label,
                # 句子
                record,
                # 语义信息
                semantic_info
            ])

    sys.stderr.write('其中异常对话: %d\n' % num_of_exception_dialogues)
    sys.stderr.write('去除异常对话后,有: %d\n' % (len(set(log_data['SessionID'])) - num_of_exception_dialogues))
    sys.stderr.write('其中\n\t全部 话语: %d\n' % num_of_sentences)
    sys.stderr.write('\tCh2R 话语: %d\n' % num_of_ch2r_sentences)
    sys.stderr.write('\tOOD 话语: %d\n' % num_of_ood_sentences)

    sys.stderr.write('\tID 话语: %d\n' % num_of_id_sentences)
    data = pd.DataFrame(
        data=datas,
        columns=[u'Sentence_ID', u'Label', u'Sentence', u'Semantic_Info']
    )
    if remove_redundant:
        sys.stderr.write('\t去重后 ID 话语: %d\n' % (len(data)))
    temp_file_path = path.join(temp_root_path,
                               'id_sentences_%s_%dsentences.csv' % (
                                   '去重' if remove_redundant else '未去重',
                                   len(data))
                               )
    write_dataframe_to_csv_file(data,
                                temp_file_path,
                                )
    sys.stderr.write('ID Sentences 缓存到: %s\n' % temp_file_path)
    return data


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


def write_dataframe_to_csv_file(data, file_path):
    """
    将 DataFrame 数据对象 写入文件中

    Parameters
    ----------
    data: pd.DataFrame()
        数据
    file_path: str
        导出文件名

    Returns
    -------

    """
    data.to_csv(
        file_path,
        sep='\t',
        encoding='utf8'
    )
