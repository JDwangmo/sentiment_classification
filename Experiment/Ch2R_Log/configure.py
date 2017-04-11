# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2017-04-06'; 'last updated date: 2017-04-06'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function
from os import path
import os

__version__ = '1.0'

Package_Root_Path = '/home/jdwang/PycharmProjects/semantic_classification/Experiment/Ch2R_Log'
# 数据文件夹
Data_Root_Path = '/home/jdwang/PycharmProjects/semantic_classification/dataset/Ch2R_Log_20150614_To_20150615/Data'
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

Verbose = 0
Option = 'rf_classifier'
# ood
# id
# id_ood : ID 加上 OOD 混合模型
Data_Type = 'id'
# boc_rule
# bow_rule
# boc
# bow
# rule
Feature_Type = 'boc_rule'
