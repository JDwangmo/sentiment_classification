# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2017-03-27'; 'last updated date: 2017-03-27'
    Email:   '383287471@qq.com'
    Describe: 请在 configure.py 中设置 变量
"""
from __future__ import print_function
import numpy as np
import sys
from os import path
import re
import codecs
from data_processing_util.feature_encoder.bow_feature_encoder import FeatureEncoder as Bow_FeatureEncoder
from data_processing_util.jiebanlp.jieba_util import Jieba_Util

__version__ = '1.0'


class FeatureEncoder(object):
    """
    ID 话语的 特征编码器
        - 陈述句特征: 3个 
        
    """

    def __init__(self,
                 feature_type='bow_rule',
                 index_to_label_name=None,
                 history_length=0,
                 temp_root_path='',
                 verbose=0
                 ):
        """
        初始化各种规则

        Parameters
        ----------
        feature_type : str
            特征类型: [ 'bow', 'rule', 'bow_rule' ]
                - bow /boc: BOW/BOC 特征
                - rule: 语法语义特征
                - bow_rule/ boc_rule: BOW / BOC 特征 + 语法语义特征
        history_length: int
            历史长度
        temp_root_path : str
            缓存文件夹
        verbose : int
            数值越大,内容越详细
        """
        # 特征类型
        self.feature_type = feature_type
        self.history_length = history_length
        self.verbose = verbose

        # 缓存文件夹
        self.temp_root_path = temp_root_path

        # 陈述句规则
        self.declarative_sentence_rules = None
        # 这些规则是 特指问 的句型模式正则
        self.definite_question_rules = None
        # 正反问
        self.positive_or_negative_question_rules = None
        # 选择问
        self.select_question_rules = None
        # 是非问
        self.yes_or_no_question_rules = None
        # 属性值列表
        self.attribute_name_count_dict = {}
        # index_to_label_name
        if index_to_label_name is None:
            self.index_to_label_name = [u'陈述句',
                                        u'特指问',
                                        u'是非问',
                                        u'正反问',
                                        u'选择问',
                                        ]
        else:
            self.index_to_label_name = index_to_label_name
        # label_name_to_index
        self.label_name_to_index = {v: k for k, v in enumerate(self.index_to_label_name)}

        self.jieba_util = Jieba_Util(verbose=verbose)

        self.feature_encoder = None

        self.sentence_to_features_temp_dict = {}

    def fit(self, train_data):
        """
        
        Parameters
        ----------
        train_data: pd.DataFrame()
            训练数据, 4个字段 : (Sentence_ID, Label, Sentence, Semantic_Info)

        Returns
        -------
        self: FeatureEncoder

        """
        id_data_train, ood_data_train = train_data
        # 检查参数
        if self.history_length > 0 and ood_data_train is not None:
            raise Exception('OOD数据没办法使用历史上下文,请只是使用 ID数据 或者 history_length设置为0\n')
        # print(train_data.describe())

        # region 1 定义各种句型模式 语法规则
        # region 这些规则是 陈述句 的句型模式正则 - 匹配上就是0 - 4个特征
        self.declarative_sentence_rules = [
            # *** + 吗/呢 + ***
            u'.*(?=吗|呢).*',
            # *** + 么
            u'.*(?=么)',
            # *** + ？(中文问号)
            u'.*？',
            # *** + ?(英文文问号)
            u'.*\\?'
        ]
        # endregion
        # region 这些规则是 特指问 的句型模式正则 - 匹配上就是1 - 5个特征
        self.definite_question_rules = [
            # ##### 规则 - 1 #####
            # *** + (是|有) + (什么|哪|多少|几) + ***
            # 比如：“价格是多少”
            u'.*(?:是|有)(?=什么|哪|多少|几).*',
            # ###################
            # ##### 规则 - 2 #####
            # 含有 (哪|什么|多少|多大|多高|几|何) 这些词，就是 特指问
            # 比如：”价格多少“、”多少钱“
            u'(?=哪|什么|多(少|大|长|高)).*',
            # ###################
            # u'.*NullKey.*(?=什么|哪|怎样|怎么样|如何|多少|几多|多).*',
            # by wjd: 其实建议这里 关键词 “何” 改为 更加准确的 “如何”
            u'.*(?=怎样|怎么样|如何|几多|多|几|何).*',
            u'.*(?=怎么|为什么|干嘛|为何).*',
            u'.*(?=什么时间|什么时候|几时|何时|哪个时候|哪个时间|几点).*',
        ]
        # endregion
        # region 这些规则是 选择问 的句型模式正则 - 匹配上就是1 - 3个特征
        self.select_question_rules = [
            # ##### 规则 - 1 #####
            # - 哪 *:
            #   - "和魅族MX4比较哪一部更加好"
            u'哪',
            # ###################
            # ##### 规则 - 2 #####
            # - 还 *:
            #   - "努比亚X6还有华为G7-TL00"
            #   - "LA3的好还是LA6的好"
            u'还',
            # ###################
            # ##### 规则 - 3 #####
            # User : 努比亚X6性能上要比华为G7-TL00好嘛？
            u'比',
            # ###################
        ]
        # endregion
        # region 这些规则是 是非问 的句型模式正则 - 匹配上就是1 - 3个特征
        self.yes_or_no_question_rules = [
            # ##### 规则 - 1
            u'.*(?=是|有).*(?=吗|么).*',
            # #####
            # ##### 规则 - 2
            u'.*(?=么)',
            # #####
            # ##### 规则 - 3
            u'.*(?=吗).*',
            # #####
        ]
        # endregion
        # region 这些规则是 正反问 的句型模式正则 - 匹配上就是1 - 4个特征
        self.positive_or_negative_question_rules = [
            # ##### 规则 - 1 #####
            # u'.*(?=有没有|是不是).*KeyValue{1,}.*',
            # u'.*KeyValue{1,}.*(?=有没有|是不是).*',
            u'.*(?=有没有|是不是).*',
            # ###################
            # ##### 规则 - 2 #####
            u'.*(?=好不好|能不能).*',
            # ###################
            # ##### 规则 - 3 #####
            # User : 还有没价格更高的
            u'.*(?=有|是|能).*(?=没有|没|不是|不|不能).*',
            # ###################
            # ##### 规则 - 4 #####
            # 匹配类似 “A* 不 A**”（ 即 前后的两个词 的前缀一定是一样的）
            # 比如 "这手机O不OK" , "便不便宜"
            u'(?P<att>.+)不(?P=att)',
            # ###################
        ]

        # endregion
        # endregion
        # region 2 定义 BOW 特征器 并训练
        self.feature_encoder = Bow_FeatureEncoder(
            verbose=0,
            need_segmented=True,
            full_mode=False,
            replace_number=True,
            remove_stopword=True,
            lowercase=True,
            feature_type='seg' if self.feature_type in ['bow', 'bow_rule'] else 'word',
            zhs2zht=True,
            remove_url=True,
            feature_method='bow',
            max_features=2000,
            word2vec_to_solve_oov=False,
        )
        sentences = []
        if id_data_train is not None:
            sentences += [utt.sentence for utt in id_data_train.iter_utterances(data_type='id')]
        if ood_data_train is not None:
            sentences += ood_data_train['Sentence'].tolist()
        self.feature_encoder.fit(
            train_data=sentences,
        )
        # endregion
        return self

    def sentence_to_features(self, sentence, semantic_info):
        """ 单句 特征

        Parameters
        ----------
        item: list
            当前话语
        data : pd.DataFrame()
            全部数据
        Returns
        -------
        feature: list
            特征
        """
        features = []

        # region 语义语义特征
        if re.search('rule', self.feature_type):
            # 取最后一句,比如:
            # "之前那个吗？　我说得是RAM容量"
            # 句型模式就根据后面那句判断就好
            # 标点符号的话,英文句号 "." 不作为切割句子的标点符号,因为 很多数字都是这样分割的比如: 5.0寸
            sentence = self.jieba_util.cut_sentence(sentence, punt_list=u'!?:;~。！？：；～')[-1]
            # print(sentence)
            # region 语法特征 - 19个
            # region 陈述句 - 4个
            for rule in self.declarative_sentence_rules:
                re_result = re.search(rule, sentence)
                if re_result:
                    # print(re_result.group())
                    features.append(0)
                else:
                    # 不匹配上为 1
                    features.append(1)
            # endregion
            # region 特指问 - 5个
            for rule in self.definite_question_rules:
                re_result = re.search(rule, sentence)
                if re_result:
                    # 匹配上为 1
                    # print(re_result.group())
                    features.append(1)
                else:
                    features.append(0)
            # endregion
            # region 选择问 - 3个
            # 3个 语法特征
            for rule in self.select_question_rules:
                re_result = re.search(rule, sentence)
                if re_result:
                    # 匹配上为 1
                    # print(re_result.group())
                    features.append(1)
                else:
                    features.append(0)

            # endregion
            # region 是非问 - 3个
            for rule in self.yes_or_no_question_rules:
                re_result = re.search(rule, sentence)
                if re_result:
                    # 匹配上为 1
                    # print(re_result.group())
                    features.append(1)
                else:
                    features.append(0)

            # endregion
            # region 正反问 - 4个
            for rule in self.positive_or_negative_question_rules:
                re_result = re.search(rule, sentence)
                if re_result:
                    # 匹配上为 1
                    # print(re_result.group())
                    features.append(1)
                else:
                    features.append(0)
            # endregion
            # endregion
            # region 语义特征 - 2个 - 是否有 值未定 的属性; 属性值最大是否>=2
            has_value = False
            # 出现过的最大属性值个数
            attribute_value_max_count = 0
            for info_block in semantic_info.split(u';'):
                # 品牌:【努比亚】;型号:【X6】
                info_block = info_block.strip()
                if len(info_block) == 0:
                    # 空,跳过
                    continue
                # 主屏尺寸:【5,2147483647】【5,2147483647】
                temps = re.split(u'[：:]', info_block)
                try:
                    attribute_name = temps[0].strip()
                    # print(attribute_name)
                    self.attribute_name_count_dict[attribute_name] = self.attribute_name_count_dict.get(attribute_name,
                                                                                                        0) + 1
                    if len(temps) == 1:
                        # 说明 只有属性名
                        # 进入这里是因为 标注的问题
                        # 比如:   价格   --> 忘记用 ":" 分割了
                        continue

                    attribute_value = temps[1].strip()
                    if len(attribute_value) == 0:
                        # print(sentence)
                        continue
                    has_value = True
                    # 属性值个数
                    attribute_value_count = max(len(attribute_value.split(u'【')) - 1, 1)
                    # print(attribute_value_count)
                    if attribute_value_count > attribute_value_max_count:
                        attribute_value_max_count = attribute_value_count
                except Exception as e:
                    # 居然出现意外?
                    sys.stderr.write(u'%s,%s,%s\n' % (sentence, semantic_info, e))
            features.append(int(has_value))
            features.append(int(attribute_value_max_count >= 2))
            # endregion
        # endregion

        # region BOW 特征
        if re.search('bo[cw]', self.feature_type):
            features += self.feature_encoder.transform_sentence(sentence).tolist()
        # endregion

        return features

    def get_features(self, item, data=None, data_type='ood'):
        """ 获取特征接口

        Parameters
        ----------
        item: list or Utterance
            当前话语
        data : pd.DataFrame()
            全部数据
        Returns
        -------
        feature: list
            特征
        """
        if data_type == 'ood':
            sentence_id, sentence = item
            semantic_info = ''
        else:
            # id
            sentence_id = item.sentence_index
            sentence = item.sentence
            semantic_info = item.semantic_info

        features = self.sentence_to_features(sentence, semantic_info)

        # 保存当前句的句子特征
        self.sentence_to_features_temp_dict[sentence_id] = features
        if data_type != 'ood' and not item.is_id:
            # 对于 ID 数据集中 非ID话语 就不返回特征了,直接 返回None
            return

        # region 对话历史特征 - 变成矩阵
        # (self.history_length + 1) * len(features)
        # (2+1) * 21 = (历史长度+当前句) * 句子特征长度
        region_encoding = np.zeros((self.history_length + 1, len(features)), dtype=np.int32)
        # print(region_encoding.shape)
        region_encoding[self.history_length] = features
        cur_sentence = item
        for history_id in reversed(np.arange(0, self.history_length, 1)):
            cur_sentence = cur_sentence.last_sentence
            if cur_sentence is None:
                continue
            if cur_sentence.sentence_index in self.sentence_to_features_temp_dict:
                history_sentence_features = self.sentence_to_features_temp_dict[cur_sentence.sentence_index]
            else:
                # 还没出现过
                history_sentence_features = self.sentence_to_features(cur_sentence.sentence, cur_sentence.semantic_info)
                # 缓存的句子特征
                self.sentence_to_features_temp_dict[cur_sentence.sentence_index] = history_sentence_features
            # print(len(history_sentence_features))

            region_encoding[history_id] = history_sentence_features
        # print(history_id, onehot_encoding)
        # features = region_encoding.flatten().tolist()
        features = region_encoding

        # endregion

        # print(features)
        # quit()
        return features

    def transform(self, data):
        """
    
        Parameters
        ----------
        data: pd.DataFrame()
            训练数据, 4个字段 : (Sentence_ID, Label, Sentence, Semantic_Info)
    
        Returns
        -------
        self: FeatureEncoder
    
        """
        # print(self.get_features([u'11', u'apple？', u'品牌:【apple】', u''],data))
        # quit()

        id_data, ood_data = data
        train_x, train_y = [], []
        if id_data is not None:
            train_x += [item
                        for item in map(lambda x: self.get_features(x, data_type='id', data=id_data),
                                        id_data.iter_utterances(data_type='id')) if item is not None
                        ]
            train_y += [item.sentence_mode for item in id_data.iter_utterances(data_type='id')]
        if ood_data is not None:
            train_x += map(lambda x: self.get_features(x, data_type='ood'),
                           ood_data[['Sentence_ID', 'Sentence']].values)
            train_y += ood_data['Label'].tolist()

        train_x = np.asarray(train_x)
        train_y = np.asarray(map(lambda x: self.label_name_to_index[x], train_y))
        # quit()
        # region 缓存文件
        # 输出到缓存文件 - attribute_names.csv
        temp_file_path = path.join(self.temp_root_path, 'attribute_names.csv')
        with codecs.open(temp_file_path, 'w', encoding='utf8') as fout:
            fout.write(u'\n'.join([u'%s\t%s' % (k, v) for k, v in self.attribute_name_count_dict.items()]))
            sys.stderr.write('%d个属性名列表 保存到缓存文件: %s\n' % (len(self.attribute_name_count_dict), temp_file_path))
        # 输出到缓存文件 - features.csv
        temp_file_path = path.join(self.temp_root_path, 'features.csv')
        # features = np.concatenate((train_y.reshape(-1, 1), train_x), axis=1)
        # features = train_x
        # np.savetxt(temp_file_path,
        #            features,
        #            delimiter=',',
        #            fmt='%d',
        #            # header='%d*%d*%d\t语法特征-19个 + 语义特征-2个' % features.shape
        #            )
        # sys.stderr.write('特征文件 保存到缓存文件: %s\n' % temp_file_path)
        # endregion

        sys.stderr.write(
            '总共有: %d 个样例, %d个对话长度(含当前句), %d 个特征, %d 个类别\n' % (
                train_x.shape[0],
                train_x.shape[1],
                train_x.shape[2],
                len(self.index_to_label_name)))
        return train_x, train_y

    def fit_transform(self, data):
        return self.fit(data).transform(data)
