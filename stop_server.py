#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行 BERT NER Server
#@Time    : 2019/1/26 21:00
# @Author  : MaCan (ma_cancan@163.com)
# @File    : run.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys


#sys.path.append('./run_script')
sys.path.append('..')

def start_server():
    from messageDisMiddle.server import MessageServer
    from messageDisMiddle.server.helper import get_run_args

    args = get_run_args()
    print(args)
    server = MessageServer(args)
    server.start()
    server.join()

def stop_server():
    from messageDisMiddle.server import MessageServer
    from messageDisMiddle.server.helper import get_run_args, get_shutdown_parser
    MessageServer.shutdown(get_shutdown_parser().parse_args())

def start_client():
    pass


# def train_ner():
#     from bert_lstm_ner import main
#     args =
#     main()


if __name__ == '__main__':
    stop_server()