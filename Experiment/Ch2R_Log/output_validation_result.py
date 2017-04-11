# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-20'
    Email:   '383287471@qq.com'
    Describe: 验证的终端输出 数据提取 脚本的 模板，分三步分别提取 参数、测试结果、训练结果
"""
from configure import Data_Type, Feature_Type

for step in [1, 2, 3, 4]:
    # step = 2
    count = 0
    with open('result/结果-20170411/%s.rf.%s.cv_result.txt' % (Data_Type, Feature_Type), 'r') as fout:
        for line in fout:
            line = line.strip()
            if step == 1:
                if line.startswith('n_estimators is '):
                    print(line.replace('n_estimators is ', '').replace('.', ''))
            if step == 2:
                if line.startswith('测试结果汇总：'):
                    print(line.replace('测试结果汇总：[', '').replace(']', ''))
            if step == 3:
                if line.startswith('验证中训练数据结果：'):
                    print(line.replace('验证中训练数据结果：[', '').replace(']', ''))
            if step == 4:
                if line.startswith('正确的个数:'):
                    # 测试集的正确个数
                    if count%8 ==1:
                        print(line.replace('正确的个数:', ''))
                    count+=1
