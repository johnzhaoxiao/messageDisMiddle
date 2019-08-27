# -*- coding: utf-8 -*-
# @Time    : 2019/3/13 10:27
# @Author  : Mat
# @Email   : 18166034717@163.com
# @File    : run_client.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import time
sys.path.append('..')
from messageDisMiddle.client import MessageClient


def ner_test_1():
    with MessageClient(ip='127.0.0.1',port=5555, port_out=5556, show_server_config=False, check_version=False, check_length=False,timeout=-1) as mc:

        # 客户端实体状态信息
        status = mc.status
        print('MessageClient status:',status)

        # 服务端实体状态信息
        server_status = mc.server_status
        print('MessageServer status:',server_status)

        # 测试预测任务
        # v1  general
        start_t = time.perf_counter()
        sents = ["原审被告人李远军，男，1981年11月13日出生，汉族，出生地湖南省新化县，文化程度中专，住广东省清远市技工学校宿舍。",
                    "被告人杨自健，男，1981年8月18日出生，汉族，出生地广东省广州市，文化程度大学，无业，系广州鹏昊兆业贸易有限公司法定代表人、总经理，住广州市白云区新科杨苑街十二巷5号。",
                    "被告人徐自忠，男，1974年12月6日出生，汉族，出生地广东省广州市，文化程度小学，住广州市白云区钟落潭镇竹料管理区小罗村。",
                    "被告人夏小云，女，1983年12月5日出生于浙江省文成县，汉族，文化程度初中，户籍地在浙江省文成县巨屿镇绸泛村垟心。"]

        para_dict = {   'model_proxy_path':'/workspace/devspace/messageDisMiddle/messageDisMiddle/model_proxy_file/ner/ner_model.proxy',
                        'pb_model_path':'/workspace/devspace/messageDisMiddle/messageDisMiddle/model_proxy_file/ner/BA/frozen_model.pb',
                        'c_tag':'BA',
                        'is_independent':False,
                        'word2id_path':'/workspace/devspace/messageDisMiddle/messageDisMiddle/model_proxy_file/ner/word2id.pkl',
        }
        # 处理消息的入口
        rst1 = mc.message_handle(sents*10,para_dict)

        print('rst:', rst1)

        print('time use:',time.perf_counter() - start_t)

#def ner_test_2():
#    with MessageClient(ip='172.17.5.164', show_server_config=False, check_version=False, check_length=False, mode='NER') as mc:


    #
if __name__ == '__main__':
    ner_test_1()
