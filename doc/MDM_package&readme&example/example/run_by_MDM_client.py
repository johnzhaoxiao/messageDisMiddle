# -*- coding: utf-8 -*-
# @Time    : 2019/3/13 10:27
# @Author  : Mat
# @Email   : 18166034717@163.com
# @File    : run_client.py
# @Software: PyCharm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from messageDisMiddle.client import MessageClient

def ner_test_1():
    with MessageClient(ip='172.17.6.232',port=10002, port_out=10003, show_server_config=False, check_version=False, check_length=False,timeout=-1) as mc:

        # 客户端实体状态信息
        status = mc.status
        print('MessageClient status:',status)

        # 服务端实体状态信息
        server_status = mc.server_status
        print('MessageServer status:',server_status)

        # 评测 NER 任务 ， entity：出生日期
        # v1  general

        sents = ["原审被告人李远军，男，1981年11月13日出生，汉族，出生地湖南省新化县，文化程度中专，住广东省清远市技工学校宿舍。",
                    "被告人杨自健，男，1981年8月18日出生，汉族，出生地广东省广州市，文化程度大学，无业，系广州鹏昊兆业贸易有限公司法定代表人、总经理，住广州市白云区新科杨苑街十二巷5号。",
                    "被告人徐自忠，男，1974年12月6日出生，汉族，出生地广东省广州市，文化程度小学，住广州市白云区钟落潭镇竹料管理区小罗村。",
                    "被告人夏小云，女，1983年12月5日出生于浙江省文成县，汉族，文化程度初中，户籍地在浙江省文成县巨屿镇绸泛村垟心。"]

        para_dict = {'model_proxy_path':'./ner_model.proxy',
                     'pb_model_path':'./BD_frozen_model.pb',
                     'c_tag':'BD',
                     'is_independent':False,
                     'word2id_path':'./word2id.pkl',
        }

        start_t = time.perf_counter()
        rst = mc.message_handle(sents*100,para_dict)
        tu = time.perf_counter() - start_t
        print(rst.tolist())
        print(tu)


def ner_test_2():
    with MessageClient(ip='172.17.6.232', port=10002, port_out=10003, show_server_config=False, check_version=False,
                       check_length=False, timeout=-1) as mc:
        # 客户端实体状态信息
        status = mc.status
        print('MessageClient status:', status)

        # 服务端实体状态信息
        server_status = mc.server_status
        print('MessageServer status:', server_status)

        # 测试预测任务
        # v1  general

        sents = ["原审被告人李远军，男，1981年11月13日出生，汉族，出生地湖南省新化县，文化程度中专，住广东省清远市技工学校宿舍。",
                 "被告人杨自健，男，1981年8月18日出生，汉族，出生地广东省广州市，文化程度大学，无业，系广州鹏昊兆业贸易有限公司法定代表人、总经理，住广州市白云区新科杨苑街十二巷5号。",
                 "被告人徐自忠，男，1974年12月6日出生，汉族，出生地广东省广州市，文化程度小学，住广州市白云区钟落潭镇竹料管理区小罗村。",
                 "被告人夏小云，女，1983年12月5日出生于浙江省文成县，汉族，文化程度初中，户籍地在浙江省文成县巨屿镇绸泛村垟心。"]

        para_dict = {'model_proxy_path': './ner_model.proxy',
                     'pb_model_path': './BD_frozen_model.pb',
                     'c_tag': 'BD',
                     'is_independent': False,
                     'word2id_path': './word2id.pkl',
                     }

        # 处理消息的入口
        data_all = []
        tu_all = []
        for i in range(1,21):
            start_t = time.perf_counter()
            rst = mc.message_handle(sents*100*i,para_dict)
            tu = time.perf_counter() - start_t
            data_all.append(4*i*100)
            tu_all.append(tu)

        #print('rst:', rst1.tolist())

    import matplotlib.pyplot as plt

    # 设置x,y轴的数值
    x = data_all
    y = tu_all
    # 创建绘图对象，figsize参数可以指定绘图对象的宽度和高度，单位为英寸，一英寸=80px
    plt.figure(figsize=(8, 4))

    # 在当前绘图对象中画图（x轴,y轴,给所绘制的曲线的名字，画线颜色，画线宽度）

    plt.plot(x, y, label="$dm=0,ds=8$", color="red", linewidth=2)
    # X轴的文字
    plt.xlabel("Data_sum(s)")

    # Y轴的文字
    plt.ylabel("Time_use")

    # 图表的标题
    plt.title("data use test")

    # Y轴的范围
    plt.ylim(0, 50)

    # 显示图示
    plt.legend()

    # 显示图
    plt.show()

    #
if __name__ == '__main__':
    ner_test_1()
