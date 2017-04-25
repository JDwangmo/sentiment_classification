# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2017-04-08'; 'last updated date: 2017-04-08'
    Email:   '383287471@qq.com'
    Describe:
"""
from __future__ import print_function
from RF_classifier import run_rf_classifier
from LSTM_classifier import run_lstm_classifier
from configure import Option
__version__ = '1.0'

if Option == 'rf_classifier':
    run_rf_classifier()
if Option == 'lstm_classifier':
    run_lstm_classifier()