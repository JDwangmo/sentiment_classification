# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-30'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function
import numpy as np
import pandas as pd
import re
from data_processing_util.cross_validation_util import get_splitted_k_fold_data_index

class DataUtil(object):
    def __init__(self):
        self.project_file = '/home/jdwang/PycharmProjects/sentiment_classification/'
        self.word2vec_model_root_path = '/home/jdwang/PycharmProjects/corprocessor/word2vec/vector/'


    def clean_str(self,string):
        """
            处理/清理文本

        """
        # 过滤字符，只保留 字母 数字 ( ) , ! ? ' `
        string = re.sub(r'[^A-Za-z0-9(),!?\'`]', ' ', string)
        # 将所有is/was等的缩写 's 在前面增加一个空格
        string = re.sub(r'\'s', ' \'s', string)
        # 将所有have等的缩写 've 在前面增加一个空格
        string = re.sub(r'\'ve', ' \'ve', string)

        # 将所有not等的缩写 n't 在前面增加一个空格 ,比如  don't --> do n't
        string = re.sub(r'n\'t', ' n\'t', string)

        # 将所有are等的缩写 're 在前面增加一个空格 ,比如  you're --> you 're
        string = re.sub(r'\'re', ' \'re', string)

        # 将所有had等的缩写 'd 在前面增加一个空格 ,比如  you'd --> you 'd
        string = re.sub(r'\'d', ' \'d', string)

        # 将所有had等的缩写 'd 在前面增加一个空格 ,比如  you'll --> you 'll
        string = re.sub(r'\'ll', ' \'ll', string)
        # 将下面标点符号两边加上空格
        string = re.sub(r',', ' , ', string)
        string = re.sub(r'!', ' ! ', string)
        string = re.sub(r'\(', ' \( ', string)
        string = re.sub(r'\)', ' \) ', string)
        string = re.sub(r'\?', ' \? ', string)

        # 将两个空白符以上的空白用一个空格替换,比如  'new   conan   and' --> 'new conna and'
        string = re.sub(r"\s{2,}", " ", string)
        # 小写
        return string.strip().lower()

    def transform_word2vec_model_name(self,flag):
        '''
            根据 flag 转换成完整的 word2vec 模型文件名

        :param flag:
        :return:
        '''

        from data_processing_util.word2vec_util.word2vec_util import Word2vecUtil
        w2v_util = Word2vecUtil()

        return w2v_util.transform_word2vec_model_name(flag)

    def load_data(self, path=None, read_type='txt'):
        """
            从文件中加载数据

        :param path: 文件路径
        :param read_type: csv,txt
        :return: texts
        :rtype: list
        """
        texts= []
        with open(path,'r') as fin:
            for line in fin:
                line = line.strip()
                texts.append(line)

        return texts

    def get_mr_dataset(self,clean_data=True,cv = 3):
        """
            获取 MR 数据集
        :param cv: 进行 cv 交叉验证 ，分数据
        :param clean_data: 是否处理数据
        :type clean_data: bool
        :return:
        """

        # 5331 positive snippets
        pos_polarity_file_path = self.project_file + 'dataset/1-MR/rt-polaritydata/rt-polarity.pos.utf8'
        # 5331 negative snippets
        neg_polarity_file_path = self.project_file + 'dataset/1-MR/rt-polaritydata/rt-polarity.neg.utf8'

        pos_texts = self.load_data(pos_polarity_file_path,read_type='txt')
        neg_texts = self.load_data(neg_polarity_file_path,read_type='txt')
        texts = pos_texts+neg_texts
        if clean_data:
            texts = map(self.clean_str,texts)

        labels = [1]*len(pos_texts) + [0]*len(neg_texts)
        # 按类别将数据分为10份，将数据 都标上 份数索引
        cv_index = get_splitted_k_fold_data_index(
            k=cv,
            data=(texts,labels),
            rand_seed=0,
        )
        data = pd.DataFrame(data={
            'TEXT':texts,
            'LABEL':labels,
            'CV_INDEX': cv_index,
        })

        return data

    def get_train_test_data(self, version='MR'):
        print('The version of data set is : %s' % version)
        if version == 'MR':
            return self.get_mr_dataset(clean_data=True)
        else:
            raise NotImplementedError



if __name__ == '__main__':
    dutil = DataUtil()
    dutil.get_train_test_data(version='MR')