# -*- coding: utf-8 -*-
# @Time    : 2019/5/16 11:15
# @Author  : Mat
# @Email   : 18166034717@163.com
# @File    : test_load_model_proxy.py
# @Software: PyCharm
import os
import sys
import pickle
import argparse
sys.path.append('../run_script')

parser = argparse.ArgumentParser(description='generate model proxy')
parser.add_argument('--mode', type=str, default='dump', help='model proxy dump or load test')
parser.add_argument('--model_proxy_path', type=str, default='./model.proxy', help='train data source')
args = parser.parse_args()
if __name__ == '__main__':

    model_proxy = args.model_proxy_path

    # dump model
    if args.mode == 'dump':
        from ner_model import NER_Model
        ner_model = NER_Model('ner')
        ner_model.dump(model_proxy)
        
    elif args.mode == 'test':
        # load model
        with open(model_proxy,'rb') as pr:
            ner_model = pickle.load(pr)

        def import_tf(device_id=-1, verbose=False, use_fp16=False):
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if device_id < 0 else str(device_id)
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if verbose else '3'
            os.environ['TF_FP16_MATMUL_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
            os.environ['TF_FP16_CONV_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
            import tensorflow as tf
            tf.logging.set_verbosity(tf.logging.DEBUG if verbose else tf.logging.ERROR)
            return tf

        sents = ["原审被告人李远军，男，1981年11月13日出生，汉族，出生地湖南省新化县，文化程度中专，住广东省清远市技工学校宿舍。",
                    "被告人杨自健，男，1981年8月18日出生，汉族，出生地广东省广州市，文化程度大学，无业，系广州鹏昊兆业贸易有限公司法定代表人、总经理，住广州市白云区新科杨苑街十二巷5号。",
                    "被告人徐自忠，男，1974年12月6日出生，汉族，出生地广东省广州市，文化程度小学，住广州市白云区钟落潭镇竹料管理区小罗村。",
                    "被告人夏小云，女，1983年12月5日出生于浙江省文成县，汉族，文化程度初中，户籍地在浙江省文成县巨屿镇绸泛村垟心。"]
        para_dict = {
                        'pb_model_path':'../model_proxy_file/ner/BA/frozen_model.pb',
                        'c_tag':'BA',
                        'is_independent':False,
                        'word2id_path':'../model_proxy_file/ner/word2id.pkl',
                        'tf':import_tf()
        }
        entity_list = ner_model.inference( sents,para_dict)
        print(entity_list)
        
    else:
        print('do nothing!')