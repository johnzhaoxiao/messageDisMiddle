# -*- coding: utf-8 -*-
# @Time    : 2019/5/15 14:58
# @Author  : Mat
# @Email   : 18166034717@163.com
# @File    : ner_model.py
# @Software: PyCharm
import os
import time
import random
import pickle
from model import Model


class NER_Model(Model):

    def __init__(self, model_name):
        super().__init__(model_name)
        #self.model_name = model_name
        self.batch_size = 128

    def dump(self, model_proxy_path):
        self.model_obj = self.__class__(self.model_name)
        with open(model_proxy_path, 'wb') as jar:
            pickle.dump(self.model_obj, jar)

    def inference(self, *args):
        '''
        args[0] = contents_list (data_list)
        args[1] = para_dict
        contents, pb_file, c_tag, is_independent, word2id_path,  tf
        :param contents: 包含序列，tag name, 预测目标是否是独立字符
        :param client_id: 客户端唯一ID
        :return:
        '''
        # from tensorflow.contrib.crf import viterbi_decode
        # from tensorflow.python.framework import graph_util
        # from tensorflow.python import pywrap_tensorflow

        ## Session configuration
        # os.environ['CUDA_VISIBLE_DEVICES'] = self.device_id
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0
        tf = args[1]['tf']
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.1  # need ~700MB GPU memory

        # list seq to be predicted
        sentents = args[0]

        # 标签str
        c_tag = args[1]['c_tag']

        # is_independent_tag
        if not args[1]['is_independent']:
            # print('adfsfsdf')
            tag2label = {"O": 0, "B-" + c_tag: 1, "I-" + c_tag: 2}
        else:
            tag2label = {"O": 0, "S-" + c_tag: 1}

        #tag2label = tag2id

        # word2id
        self.word2id = self.read_dictionary(args[1]['word2id_path'])
        #print(self.word2id)
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            with open(args[1]['pb_model_path'], 'rb') as f:
                output_graph_def.ParseFromString(f.read())
                tf.import_graph_def(output_graph_def, name="")
            with tf.Session(config=config) as sess:
                sess.run(tf.global_variables_initializer())
                # 定义输入的张量名称，对应网络结构的输入张量
                #
                input_word_ids_tensor = sess.graph.get_tensor_by_name("word_ids:0")
                input_dropout_tensor = sess.graph.get_tensor_by_name("dropout:0")
                input_sequence_lengths = sess.graph.get_tensor_by_name("sequence_lengths:0")

                # 定义输出的张量的名称
                output_logits = sess.graph.get_tensor_by_name("proj/logits:0")
                output_transition = sess.graph.get_tensor_by_name("transition_out:0")

                demo_sents = [list(sentent.strip()) for sentent in sentents]
                demo_data = [(demo_sent, ['O'] * len(demo_sent)) for demo_sent in demo_sents]

                label_list = []
                for seqs, labels in self.batch_yield(demo_data, self.batch_size, self.word2id, tag2label, shuffle=False):
                    # print(seqs)
                    # print(labels)
                    word_ids, seq_len_list = self.pad_sequences(seqs, pad_mark=0)

                    feed_dict = {input_word_ids_tensor: word_ids,
                                 input_sequence_lengths: seq_len_list,
                                 input_dropout_tensor: 1.0}
                    out_logits, out_transition = sess.run([output_logits, output_transition], feed_dict=feed_dict)
                    _label_list = []
                    for logit, seq_len in zip(out_logits, seq_len_list):
                        # from tensorflow import tf
                        viterbi_seq, _ = tf.contrib.crf.viterbi_decode(logit[:seq_len], out_transition)
                        _label_list.append(viterbi_seq)
                        # print(viterbi_seq)

                    label_list.extend(_label_list)
                    # print(label_list)
                label2tag = {}
                for tag, label in tag2label.items():
                    label2tag[label] = tag if label != 0 else label
                # print(label_list,len(label_list))
                # print(label2tag)

                tag = [[label2tag[label] for label in label_li] for label_li in label_list]
                # print(tag)
                # print(len(tag))
        ##
        entity_list = []
        for i, demo_sent in enumerate(demo_sents):
            entity = self.get_tag_entity(tag[i], demo_sent, tag2label)
            entity_list.append(entity)
        return entity_list

    def batch_yield(self, data, batch_size, vocab, tag2label, shuffle=False):
        """

        :param data:
        :param batch_size:
        :param vocab:
        :param tag2label:
        :param shuffle:
        :return:
        """
        if shuffle:
            random.shuffle(data)

        seqs, labels = [], []
        for (sent_, tag_) in data:
            sent_ = self.sentence2id(sent_, vocab)
            label_ = [tag2label[tag] for tag in tag_]

            if len(seqs) == batch_size:
                yield seqs, labels
                seqs, labels = [], []

            seqs.append(sent_)
            labels.append(label_)

        if len(seqs) != 0:
            yield seqs, labels

    def sentence2id(self, sent, word2id):
        """

        :param sent:
        :param word2id:
        :return:
        """
        sentence_id = []
        for word in sent:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word = '<UNK>'
            sentence_id.append(word2id[word])
        return sentence_id

    def pad_sequences(self, sequences, pad_mark=0):
        """

        :param sequences:
        :param pad_mark:
        :return:
        """
        max_len = max(map(lambda x: len(x), sequences))
        seq_list, seq_len_list = [], []
        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
            seq_list.append(seq_)
            seq_len_list.append(min(len(seq), max_len))
        return seq_list, seq_len_list

    def get_tag_entity(self, tag_seq, char_seq, tag2label):
        for tag in tag2label.keys():
            if tag != "O":
                tag_name = tag[2:]
        length = len(char_seq)
        entity = []
        for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
            if tag == 'B-' + tag_name:
                if 'per' in locals().keys():
                    entity.append(per)
                    del per
                per = char
                if i + 1 == length:
                    entity.append(per)
            if tag == 'I-' + tag_name:
                if 'per' not in locals().keys():
                    continue
                per += char
                if i + 1 == length:
                    entity.append(per)
            if tag not in ['I-' + tag_name, 'B-' + tag_name]:
                if 'per' in locals().keys():
                    entity.append(per)
                    del per
                continue
        return entity

    def read_dictionary(self, vocab_path):
        """

        :param vocab_path:
        :return:
        """
        vocab_path = os.path.join(vocab_path)
        with open(vocab_path, 'rb') as fr:
            word2id = pickle.load(fr)
        #print('vocab_size:', len(word2id))
        return word2id

    def __getstate__(self):
        return self.model_name

    def __setstate__(self, model_name):
        self.__init__(model_name)


if __name__ == '__main__':

    ner_model_proxy = '../model_proxy_file/ner.model'

    # dump model
    ner_model = NER_Model('ner')
    ner_model.dump(ner_model_proxy)

    # load model
    # with open(ner_model_proxy,'rb') as pr:
    #     ner_model = pickle.load(pr)
    #
    # def import_tf(device_id=-1, verbose=False, use_fp16=False):
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if device_id < 0 else str(device_id)
    #     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if verbose else '3'
    #     os.environ['TF_FP16_MATMUL_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
    #     os.environ['TF_FP16_CONV_USE_FP32_COMPUTE'] = '0' if use_fp16 else '1'
    #     import tensorflow as tf
    #     tf.logging.set_verbosity(tf.logging.DEBUG if verbose else tf.logging.ERROR)
    #     return tf
    #
    # sents = ["原审被告人李远军，男，1981年11月13日出生，汉族，出生地湖南省新化县，文化程度中专，住广东省清远市技工学校宿舍。",
    #             "被告人杨自健，男，1981年8月18日出生，汉族，出生地广东省广州市，文化程度大学，无业，系广州鹏昊兆业贸易有限公司法定代表人、总经理，住广州市白云区新科杨苑街十二巷5号。",
    #             "被告人徐自忠，男，1974年12月6日出生，汉族，出生地广东省广州市，文化程度小学，住广州市白云区钟落潭镇竹料管理区小罗村。",
    #             "被告人夏小云，女，1983年12月5日出生于浙江省文成县，汉族，文化程度初中，户籍地在浙江省文成县巨屿镇绸泛村垟心。"]
    # contents = {}
    # contents[0] = sents
    # contents[1] = 'BA'
    # contents[2] = False
    # entity_list, client_id = ner_model.inference('xxasuojblsdknf', contents, '../model_proxy_file/BA/frozen_model.pb', '../model_proxy_file/word2id.pkl',  import_tf())
    # print(entity_list)
    # print(client_id)