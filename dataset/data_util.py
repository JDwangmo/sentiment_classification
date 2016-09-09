# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-30'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function

import re

import pandas as pd


class DataUtil(object):
    def __init__(self):
        self.project_file = '/home/jdwang/PycharmProjects/sentiment_classification/'
        self.word2vec_model_root_path = '/home/jdwang/PycharmProjects/corprocessor/word2vec/vector/'

    def clean_str(self, string, version='MR',):
        """
            处理/清理文本

        Parameters
        ----------
        string
        version : str
            - MR
            - SST

        Returns
        ----------
        string : str
        """

        if version in ['MR','TREC','Subj']:
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
            return string.strip().lower() if version!='TREC' else string.strip()
        elif version == 'SST':
            string = string.replace(u'-LRB-', u'(')
            string = string.replace(u'-RRB-', u')')

            # string = re.sub(r'[^A-Za-z0-9(),!?\'`]', ' ', string)

            # region 发现直接过滤掉即可，下面全部注释掉
            # 主要处理文件 datasetSentences.txt 中的 符号
            string = string.replace(u'Â', u'')
            string = string.replace(u'Ã©', u'e')
            string = string.replace(u'Ã¨', u'e')
            string = string.replace(u'Ã¯', u'i')
            string = string.replace(u'Ã³', u'o')
            string = string.replace(u'Ã´', u'o')
            string = string.replace(u'Ã¶', u'o')
            string = string.replace(u'Ã±', u'n')
            string = string.replace(u'Ã¡', u'a')
            string = string.replace(u'Ã¢', u'a')
            string = string.replace(u'Ã£', u'a')
            string = string.replace(u'\xc3\x83\xc2\xa0', u'a')
            string = string.replace(u'Ã¼', u'u')
            string = string.replace(u'Ã»', u'u')
            string = string.replace(u'Ã§', u'c')
            string = string.replace(u'Ã¦', u'ae')
            string = string.replace(u'Ã­', u'i')
            string = string.replace(u'\xa0', u' ')
            string = string.replace(u'\xc2', u'')
            string = string.replace(u'Ã ', u'a')


            #region 主要处理文件 dictionary.txt 中的符号
            string = string.replace(u'é', u'e')
            string = string.replace(u'è', u'e')
            string = string.replace(u'ï', u'i')
            string = string.replace(u'í', u'i')
            string = string.replace(u'ó', u'o')
            string = string.replace(u'ô', u'o')
            string = string.replace(u'ö', u'o')
            string = string.replace(u'á', u'a')
            string = string.replace(u'â', u'a')
            string = string.replace(u'ã', u'a')
            string = string.replace(u'à', u'a')
            string = string.replace(u'ü', u'u')
            string = string.replace(u'û', u'u')
            string = string.replace(u'ñ', u'n')
            string = string.replace(u'ç', u'c')
            string = string.replace(u'æ', u'ae')
            string = string.replace(u'\xa0', u' ')
            string = string.replace(u'\xc2', u'')

            # endregion

            # 将两个空白符以上的空白用一个空格替换,比如  'new   conan   and' --> 'new conna and'
            string = re.sub(r"\s{2,}", " ", string)

            # 转换为小写
            # 这里把 小写转换 注释掉，放到查找完字典之后，主要是因为查找字典时 发现 有些 phrase 大写和小写的 语义是不一样的，所以，不能在前面转换为小写。比如No是negative，而no是中性。
            # string = string.strip().lower()
            return string

    def transform_word2vec_model_name(self, flag):
        """
            根据 flag 转换成完整的 word2vec 模型文件名

        :param flag:
        :return:
        """

        from data_processing_util.word2vec_util.word2vec_util import Word2vecUtil
        w2v_util = Word2vecUtil()

        return w2v_util.transform_word2vec_model_name(flag)

    def load_data(self, path=None, load_type='txt', **kwargs):
        """
            从文件中加载数据

        Parameters
        ----------
        path : str
            文件路径
        load_type :
            csv,txt
        kwargs : dict
            - sep
            - header

        Returns
        ----------
        texts : list or DataFrame
        """
        if load_type == 'txt':
            # 一句句的读进来 ，一行一个样例,返回列表
            texts = []
            with open(path, 'r') as fin:
                for line in fin:
                    line = line.strip()
                    texts.append(line)
        elif load_type == 'csv':
            # 一行有多个字段，有表头，且以 \t 分割各个字段,返回 DataFrame() object
            texts = pd.read_csv(
                path,
                encoding='utf8',
                sep=kwargs.get('sep', '\t'),
                header=kwargs.get('header', 0),
            )
        else:
            raise NotImplementedError

        return texts

    def get_mr_dataset(self, clean_data=True):
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

        pos_texts = self.load_data(pos_polarity_file_path, load_type='txt')
        neg_texts = self.load_data(neg_polarity_file_path, load_type='txt')
        texts = pos_texts + neg_texts
        if clean_data:
            texts = map(lambda x: self.clean_str(x, version='MR'), texts)

        labels = [1] * len(pos_texts) + [0] * len(neg_texts)

        data = pd.DataFrame(data={
            'TEXT': texts,
            'LABEL': labels,
        })

        return data

    def sentiment_value_to_label_index(self, sentiment_value):
        """
            [0, 0.2], (0.2, 0.4], (0.4, 0.6], (0.6, 0.8], (0.8, 1.0]
            ---> 0,1,2,3,4
        Parameters
        ----------
        sentiment_value : float

        Returns
        -------

        """
        if 0. <= sentiment_value <= 0.2:
            return 0
        elif 0.2 < sentiment_value <= 0.4:
            return 1
        elif 0.4 < sentiment_value <= 0.6:
            return 2
        elif 0.6 < sentiment_value <= 0.8:
            return 3
        elif 0.8 < sentiment_value <= 1.0:
            return 4
        else:
            raise NotImplementedError

    def get_sst_dataset(self, version='SST-1'):
        """
        # SST 数据集句子 (Stanford Sentiment Treebank V1.0)
        # 已经分为 train、test、dev 集 , SPLITSET_LABEL字段， 1 = train, 2 = test, 3 = dev

        Parameters
        ----------
        version : str
            - SST-1： 11855 samples, train/test/dev = 8544 / 1101/ 2210 , 5-labels, 0-4
            - SST-2： Same as SST-1 but with neutral reviews removed and binary labels,0 or 1，
                      9613 samples, train/test/dev = 6920 / 872/ 1821
        """

        # region 1、加载数据
        sst_file_path = self.project_file + 'dataset/2-SST/stanfordSentimentTreebank/datasetSentences.txt'
        # 分为 train、test、dev 集 ,1 = train, 2 = test, 3 = dev
        dataset_split_file_path = self.project_file + 'dataset/2-SST/stanfordSentimentTreebank/datasetSplit.txt'
        # 字典
        dictionary_file_path = self.project_file + 'dataset/2-SST/stanfordSentimentTreebank/dictionary.txt'
        # 每个 phrase 语义标签
        sentiment_labels_file_path = self.project_file + 'dataset/2-SST/stanfordSentimentTreebank/sentiment_labels.txt'
        # sentence_index | sentence
        texts = self.load_data(sst_file_path, load_type='csv')
        # sentence_index | splitset_label
        dataset_split = self.load_data(dataset_split_file_path, load_type='csv', sep=',')
        # phrase | phrase id
        dictionary = self.load_data(dictionary_file_path, load_type='csv', sep='|', header=None)
        # phrase ids | sentiment values
        sentiment_labels = self.load_data(sentiment_labels_file_path, load_type='csv', sep='|')
        # endregion
        # 对文本进行处理，统一字符
        texts['sentence'] = texts['sentence'].map(lambda x: self.clean_str(x, version='SST'))
        dictionary[0] = dictionary[0].map(lambda x: self.clean_str(x, version='SST'))

        # phrase_id 映射到 phrase
        index_2_phrase = {k: v for v, k in dictionary.values}
        # 三个字段： PHRASE  LABEL
        phrase = pd.DataFrame()
        phrase['PHRASE'] = sentiment_labels['phrase ids'].map(index_2_phrase)
        phrase['LABEL'] = sentiment_labels['sentiment values'].map(self.sentiment_value_to_label_index)

        # phrase 转为 语义label

        phrase_to_label = {k: v for k, v in phrase[['PHRASE', 'LABEL']].values}
        # 通过查字典 得到 句子 的 语义LABEL
        texts['LABEL'] = texts['sentence'].map(phrase_to_label)
        # 这里把 小写转换 放到这里，主要是因为查找字典时 发现 有些 phrase 大写和小写的 语义是不一样的，所以，不能在前面转换为小写。比如No是negative，而no是 neutral 的。
        texts['TEXT'] = texts['sentence'].apply(lambda x: re.sub(r'[^A-Za-z0-9(),!?\'`]', ' ', x))
        texts['TEXT'] = texts['TEXT'].apply(lambda x: re.sub(r"\s{2,}", " ", x))
        texts['TEXT'] = texts['TEXT'].apply(lambda x: x.strip().lower())

        texts = texts[['sentence_index', 'TEXT', 'LABEL']]
        texts['SPLITSET_LABEL'] = dataset_split['splitset_label']
        # print(texts[texts['LABEL'].isnull()])
        if version=='SST-1':
            # 五分类，fine,0-4
            # for very negative, negative, neutral, positive, very positive, respectively.
            # 11855 samples totally
            # 不用进进行任何操作
            pass
        elif version=='SST-2':
            # 去除 中性 的标签（2）的数据
            # Same as SST-1 but with neutral reviews removed and binary labels,0 or 1
            # 9613 samples totally
            texts = texts[texts['LABEL']!=2]
            # negative=0 , positive=1
            # texts.to_csv('/home/jdwang/PycharmProjects/sentiment_classification/SST/result/temp.csv',
            #              sep='\t',
            #              encoding='utf8')
            texts['LABEL'] = texts['LABEL'].apply(lambda x: int(x>2))
        else:
            raise NotImplementedError
        return texts

    def get_trec_dataset(self):
        train_data_file_path = self.project_file + 'dataset/3-TREC/train_5452.label.txt'
        test_data_file_path = self.project_file + 'dataset/3-TREC/test_500.label.txt'

        train_data = self.load_data(train_data_file_path,load_type='txt')
        test_data = self.load_data(test_data_file_path,load_type='txt')

        train_x=[]
        train_y=[]
        for item in train_data:
            items = item.split(':')
            train_y.append(items[0])
            train_x.append(':'.join(items[1:]))

        test_x=[]
        test_y=[]
        for item in test_data:
            items = item.split(':')
            test_y.append(items[0])
            test_x.append(':'.join(items[1:]))

        texts = train_x + test_x
        texts = map(lambda x: self.clean_str(x,version='TREC'),texts)
        labels = train_y+test_y

        splitset_label = [1]*len(train_x)+[2]*len(test_x)
        data = pd.DataFrame(
            data={
                'TEXT':texts,
                'LABEL':labels,
                'SPLITSET_LABEL':splitset_label,
            }
        )
        labels_to_index = {'ABBR':0,'ENTY':1, 'DESC':2,'HUM':3,'LOC':4,'NUM':5}
        data['LABEL'] = data['LABEL'].map(labels_to_index)

        return data

    def get_subj_dataset(self):
        # 5000 objective snippets
        objective_subj_file_path = self.project_file + 'dataset/5-Subj/plot.tok.gt9.5000'
        # 5000 subjective snippets
        subjective_subj_file_path = self.project_file + 'dataset/5-Subj/quote.tok.gt9.5000'

        objective_texts = self.load_data(objective_subj_file_path, load_type='txt')
        subjective_texts = self.load_data(subjective_subj_file_path, load_type='txt')
        texts = objective_texts + subjective_texts

        texts = map(lambda x: self.clean_str(x, version='Subj'), texts)

        labels = [1] * len(objective_texts) + [0] * len(subjective_texts)

        data = pd.DataFrame(data={
            'TEXT': texts,
            'LABEL': labels,
        })

        return data

    def get_train_test_data(self, version='MR'):
        """

        Parameters
        ----------
        version : str
            - MR: movie review dataset, 10662 samples ( 5331 positive and 5331 negative)
            - SST-1： 11855 samples, train/test/dev = 8544 / 1101/ 2210
            - SST-2： Same as SST-1 but with neutral reviews removed and binary labels,0 or 1，
                      9613 samples, train/test/dev = 6920 / 872/ 1821
            - TREC :
        Returns
        -------

        """
        print('The version of data set is : %s' % version)
        if version == 'MR':
            return self.get_mr_dataset(clean_data=True)
        if version == 'SST-1':
            return self.get_sst_dataset(version='SST-1')
        if version == 'SST-2':
            return self.get_sst_dataset(version='SST-2')
        if version == 'TREC':
            return self.get_trec_dataset()
        if version == 'Subj':
            return self.get_subj_dataset()
        else:
            raise NotImplementedError


def output_validation_result(path, version='CNN-A00', step=1):
    """将验证结果的文件提取出数据信息

    Parameters
    ----------
    path : str
        路径信息
    version : str,
        哪个模型版本的验证输出文件
    step : int
        步骤

    Raises
    --------
    NotImplementedError
    """

    with open(path, 'r') as fout:
        for line in fout:
            line = line.strip()
            if version == 'CNN-A00':
                if step == 1:
                    if line.startswith('num_filter is '):
                        print(line.replace('num_filter is ', '').replace('.', ''))
                if step == 2:
                    if line.startswith('测试结果汇总：'):
                        print(line.replace('测试结果汇总：[', '').replace(']', ''))
                if step == 3:
                    if line.startswith('验证中训练数据结果：'):
                        print(line.replace('验证中训练数据结果：[', '').replace(']', ''))
            else:
                raise NotImplementedError


if __name__ == '__main__':
    dutil = DataUtil()
    dutil.get_train_test_data(version='Subj')


    # output_validation_result('/home/jdwang/PycharmProjects/sentiment_classification/MR/result/MR_cnn-static-w2v_cv.txt',version='CNN-A00',step=1)
