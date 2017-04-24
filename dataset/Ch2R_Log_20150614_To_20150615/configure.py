# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2017-04-06'; 'last updated date: 2017-04-06'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function

import datetime
from os import path
import os
import sys
__version__ = '1.0'

Package_Root_Path = '/home/jdwang/PycharmProjects/semantic_classification/dataset/Ch2R_Log_20150614_To_20150615'
# 数据文件夹
Data_Root_Path = path.join(Package_Root_Path, 'Data')
# 数据文件
Data_File_Path = path.join(Data_Root_Path, 'ID标注-汇总-20170404.csv')
# 缓存文件夹
Temp_Root_Path = path.join(Package_Root_Path, 'Temp')

# 检验路径是否存在
if not path.exists(Data_Root_Path):
    raise Exception('数据文件夹路径有问题啊:%s' % Data_Root_Path)

if not path.exists(Data_File_Path):
    raise Exception('数据文件夹路径有问题啊:%s' % Data_File_Path)

if not path.exists(Temp_Root_Path):
    os.makedirs(Temp_Root_Path)

# ood
# id
# id_ood : ID 加上 OOD 混合模型
Data_Type = 'id'
# boc_rule
# bow_rule
# boc
# bow
# rule
# boc_rule_history
Feature_Type = 'rule_boc'
# 对话历史的长度
History_Length = 1

Verbose = 0

# region 打印变量设置情况
sys.stderr.write('_' * 80 + '\n')
sys.stderr.write('-' * 80 + '\n')
sys.stderr.write('现在是: %s\n' % (datetime.datetime.now()))
sys.stderr.write('变量设置情况\n')
sys.stderr.write('Data_Type: %s,\t' % Data_Type)
sys.stderr.write('Verbose: %s\n' % Verbose)
sys.stderr.write('Feature_Type: %s,\t' % Feature_Type)
sys.stderr.write('History_Length: %d\n' % History_Length)
sys.stderr.write('-' * 80 + '\n')
# endregion
