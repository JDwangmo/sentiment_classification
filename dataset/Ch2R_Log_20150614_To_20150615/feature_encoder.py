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
from configure import Temp_Root_Path
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
                 temp_root_path=Temp_Root_Path,
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
        temp_root_path : str
            缓存文件夹
        verbose : int
            数值越大,内容越详细
        """
        # 特征类型
        self.feature_type = feature_type
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
        # print(train_data.describe())
        # region 1 类别的分布情况
        if self.verbose > 0:
            print('类别的分布情况')
            print(train_data['Label'].value_counts())
        # endregion
        # region 2 定义各种句型模式 语法规则
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
        self.feature_encoder.fit(
            train_data=train_data['Sentence'].as_matrix()
        )

        return self

    def get_features(self, item):
        features = []

        sentence, semantic_info = item
        # region BOW 特征
        if self.feature_type in ['bow', 'boc', 'bow_rule', 'boc_rule']:
            features += self.feature_encoder.transform_sentence(sentence).tolist()
        # endregion
        # region 语义语义特征
        if self.feature_type in ['rule', 'bow_rule', 'boc_rule']:
            # 取最后一句,比如:
            # "之前那个吗？　我说得是RAM容量"
            # 句型模式就根据后面那句判断就好
            sentence = self.jieba_util.cut_sentence(sentence)[-1]
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
        # print(self.get_features([u'这款手机外观好不好看？', u'外观设计:【好】']))
        # quit()

        train_x = np.asarray(map(self.get_features, data[['Sentence', 'Semantic_Info']].values))

        train_y = np.asarray(data['Label'].map(self.label_name_to_index))
        # region 缓存文件
        # 输出到缓存文件 - attribute_names.csv
        temp_file_path = path.join(self.temp_root_path, 'attribute_names.csv')
        with codecs.open(temp_file_path, 'w', encoding='utf8') as fout:
            fout.write(u'\n'.join([u'%s\t%s' % (k, v) for k, v in self.attribute_name_count_dict.items()]))
            sys.stderr.write('%d个属性名列表 保存到缓存文件: %s\n' % (len(self.attribute_name_count_dict), temp_file_path))
        # 输出到缓存文件 - features.csv
        temp_file_path = path.join(self.temp_root_path, 'features.csv')
        features = np.concatenate((train_y.reshape(-1, 1), train_x), axis=1)
        np.savetxt(temp_file_path,
                   features,
                   delimiter=',',
                   fmt='%d',
                   header='%d*%d\tLabel-1个 + 语法特征-19个 + 语义特征-2个' % features.shape
                   )
        sys.stderr.write('特征文件 保存到缓存文件: %s\n' % temp_file_path)
        # endregion

        sys.stderr.write(
            '总共有: %d 个样例, %d 个特征, %d 个类别\n' % (train_x.shape[0], train_x.shape[1], len(self.index_to_label_name)))
        return train_x, train_y

    def fit_transform(self, data):
        return self.fit(data).transform(data)
