#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

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
    pass


def start_client():
    pass


# def train_ner():
#     from bert_lstm_ner import main
#     args =
#     main()


if __name__ == '__main__':
    start_server()