#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
import random
import sys
import threading
import time
import pickle
from collections import defaultdict
from datetime import datetime
from multiprocessing import Process
from multiprocessing.pool import Pool

import numpy as np
import zmq
import zmq.decorators as zmqd
from termcolor import colored
from zmq.utils import jsonapi

from .helper import *
from .http import HTTPProxy
from .zmq_decor import multi_socket

# import tensorflow as tf
# from tensorflow.contrib.crf import viterbi_decode
# import numpy as np

#from .data import read_dictionary, pad_sequences, batch_yield

# from data import read_corpus
#sys.path.append('./run_script')
# from bert_base.train.models import convert_id_to_label

__all__ = ['__version__', 'MessageServer']
__version__ = '1.0'

_tf_ver_ = check_tf_version()


class ServerCommand:
    terminate = b'TERMINATION'
    show_config = b'SHOW_CONFIG'
    new_job = b'REGISTER'

    @staticmethod
    def is_valid(cmd):
        return any(not k.startswith('__') and v == cmd for k, v in vars(ServerCommand).items())


class MessageServer(threading.Thread):
    def __init__(self, args):
        super().__init__()
        sys.path.append(args.run_script_dir)

        self.logger = set_logger(colored('VENTILATOR', 'red'), verbose=args.verbose, logfile=args.logger)
        self.logger.info('run_script_dir: %s'% args.run_script_dir)
        self.num_worker = args.num_worker
        self.max_batch_size = args.max_batch_size
        self.num_concurrent_socket = max(8, args.num_worker * 2)  # optimize concurrency for multi-clients
        self.port = args.port
        self.args = args
        self.args.device_map = self.__check_arg_device_map()
        self.status_args = {k: (v if k != 'pooling_strategy' else v.value) for k, v in sorted(vars(args).items())}
        self.status_static = {
            'tensorflow_version': _tf_ver_,
            'python_version': sys.version,
            'server_version': __version__,
            'pyzmq_version': zmq.pyzmq_version(),
            'zmq_version': zmq.zmq_version(),
            'server_start_time': str(datetime.now()),
        }
        self.processes = []
        self.is_ready = threading.Event()
        self.__set_zmq_socket_tmp_dir()

    def __check_arg_device_map(self):
        device_map = list(set(-1 if int(i) < 0 else int(i) for i in self.args.device_map.split(',')))
        if -1 in device_map:
            return [-1]
        else:
            return device_map

    def __set_zmq_socket_tmp_dir(self):
        if self.args.zeromq_sock_tmp_dir:
            os.environ['ZEROMQ_SOCK_TMP_DIR'] = self.args.zeromq_sock_tmp_dir
        else:
            if not os.path.exists('./sock_tmp'):
                os.makedirs('./sock_tmp')
            os.environ['ZEROMQ_SOCK_TMP_DIR'] = './sock_tmp'
    def close(self):
        self.logger.info('shutting down...')
        self._send_close_signal()
        for p in self.processes:
            p.close()
        self.join()

    @zmqd.context()
    @zmqd.socket(zmq.PUSH)
    def _send_close_signal(self, _, frontend):
        frontend.connect('tcp://localhost:%d' % self.port)
        frontend.send_multipart([b'', ServerCommand.terminate, b'', b''])

    @staticmethod
    def shutdown(args):
        with zmq.Context() as ctx:
            ctx.setsockopt(zmq.LINGER,args.timeout)
            with ctx.socket(zmq.PUSH) as frontend:
                try:
                    frontend.connect('tcp://%s:%d' %(args.ip, args.port))
                    frontend.send_multipart([b'0',ServerCommand.terminate, b'0', b'0'])
                    print('shutdown signal sent to %d' % args.port)
                    #os._exit(0)
                except zmq.error.Again:
                    raise TimeoutError('no response from the server (with "timeout"=%d ms), please check the following:'
                                       'is the server still online? is the network broken? are "port" correct? ' % args.timeout)


    def run(self):
        self._run()

    @zmqd.context()
    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @multi_socket(zmq.PUSH, num_socket='num_concurrent_socket')
    def _run(self, _, frontend, sink, *backend_socks):

        def push_new_job(_job_id, _json_msg, _msg_len):
            # backend_socks[0] is always at the highest priority
            _sock = backend_socks[0] if _msg_len <= self.args.priority_batch_size else rand_backend_socket
            _sock.send_multipart([_job_id, _json_msg])

        # bind all sockets
        self.logger.info(str(self.args))
        self.logger.info('bind all sockets')
        frontend.bind('tcp://*:%d' % self.port)
        addr_front2sink = auto_bind(sink)
        addr_backend_list = [auto_bind(b) for b in backend_socks]
        self.logger.info('open %d ventilator-worker sockets, %s'
                         % (len(addr_backend_list), ','.join(addr_backend_list)))

        # start the sink process
        # sink是用来接收上层MessageWork的产出，然后发送给client
        self.logger.info('start the sink')
        proc_sink = MessageSink(self.args, addr_front2sink)
        self.processes.append(proc_sink)
        proc_sink.start()
        addr_sink = sink.recv().decode('ascii')

        # start the backend processes
        # 这里启动多个进程，加载Worker处理数据和模型
        device_map = self._get_device_map()
        for idx, device_id in enumerate(device_map):

            process = MessageWorker(idx, self.args, addr_backend_list, addr_sink, device_id)
            self.processes.append(process)
            process.start()

        # start the http-service process
        if self.args.http_port:
            self.logger.info('start http proxy')
            proc_proxy = HTTPProxy(self.args)
            self.processes.append(proc_proxy)
            proc_proxy.start()

        rand_backend_socket = None
        server_status = ServerStatistic()
        while True:
            try:
                request = frontend.recv_multipart()
                client, msg, req_id, msg_len = request
                # print(request)
            except ValueError:
                self.logger.error('received a wrongly-formatted request (expected 4 frames, got %d)' % len(request))
                self.logger.error('\n'.join('field %d: %s' % (idx, k) for idx, k in enumerate(request)), exc_info=True)
            else:
                server_status.update(request)
                if msg == ServerCommand.terminate:
                    break
                elif msg == ServerCommand.show_config:
                    self.logger.info('new config request\treq id: %d\tclient: %s' % (int(req_id), client))
                    status_runtime = {'client': client.decode('ascii'),
                                      'num_process': len(self.processes),
                                      'ventilator -> worker': addr_backend_list,
                                      'worker -> sink': addr_sink,
                                      'ventilator <-> sink': addr_front2sink,
                                      'server_current_time': str(datetime.now()),
                                      'statistic': server_status.value,
                                      'device_map': device_map,
                                      'num_concurrent_socket': self.num_concurrent_socket}

                    sink.send_multipart([client, msg, jsonapi.dumps({**status_runtime,
                                                                     **self.status_args,
                                                                     **self.status_static}), req_id])
                else:
                    self.logger.info('new ner request\treq id: %d\tsize: %d\tclient: %s' %
                                     (int(req_id), int(msg_len), client))
                    # register a new job at sink
                    sink.send_multipart([client, ServerCommand.new_job, msg_len, req_id])

                    # renew the backend socket to prevent large job queueing up
                    # [0] is reserved for high priority job
                    # last used backennd shouldn't be selected either as it may be queued up already
                    rand_backend_socket = random.choice([b for b in backend_socks[1:] if b != rand_backend_socket])

                    # push a new job, note super large job will be pushed to one socket only,
                    # leaving other sockets free
                    job_id = client + b'#' + req_id
                    if int(msg_len) > self.max_batch_size:
                        seqs = jsonapi.loads(msg)
                        seq = seqs[0]
                        job_gen = ((job_id + b'@%d' % i, [seq[i:(i + self.max_batch_size)], seqs[1]]) for i in
                                   range(0, int(msg_len), self.max_batch_size))
                        for partial_job_id, job in job_gen:
                            push_new_job(partial_job_id, jsonapi.dumps(job), len(job))
                    else:
                        push_new_job(job_id, msg, int(msg_len))

        self.logger.info('terminated!')

    def _get_device_map(self):
        self.logger.info('get devices')
        run_on_gpu = False
        device_map = [-1] * self.num_worker
        if not self.args.device_map[0]==-1:
            try:
                import GPUtil
                num_all_gpu = len(GPUtil.getGPUs())
                #print('num_all_gpu:',num_all_gpu)
                avail_gpu = GPUtil.getAvailable(order='memory', limit=min(num_all_gpu, self.num_worker))
                #print(GPUtil.getAvailability(GPUtil.getGPUs(), maxLoad=0.5, maxMemory=0.5, memoryFree=0))
                #gpu_= [[g.memoryFree,g.memoryTotal] for g in GPUtil.getGPUs()]
                #print("avail_gpu:",avail_gpu)
                num_avail_gpu = len(avail_gpu)
                #print("num_avail_gpu:",num_avail_gpu)
                if num_avail_gpu >= self.num_worker:
                    run_on_gpu = True
                elif 0 < num_avail_gpu < self.num_worker:
                    self.logger.warning('only %d out of %d GPU(s) is available/free, but "-num_worker=%d"' %
                                        (num_avail_gpu, num_all_gpu, self.num_worker))
                    if not self.args.device_map:
                        self.logger.warning('multiple workers will be allocated to one GPU, '
                                            'may not scale well and may raise out-of-memory')
                    else:
                        self.logger.warning('workers will be allocated based on "-device_map=%s", '
                                            'may not scale well and may raise out-of-memory' % self.args.device_map)
                    run_on_gpu = True
                else:
                    self.logger.warning('no GPU available, fall back to CPU')

                if run_on_gpu:
                    device_map = ((self.args.device_map or avail_gpu) * self.num_worker)[: self.num_worker]
            except FileNotFoundError:
                self.logger.warning('nvidia-smi is missing, often means no gpu on this machine. '
                                    'fall back to cpu!')
        self.logger.info('device map: \n\t\t%s' % '\n\t\t'.join(
            'worker %2d -> %s' % (w_id, ('gpu %2d' % g_id) if g_id >= 0 else 'cpu') for w_id, g_id in
            enumerate(device_map)))
        return device_map


class MessageSink(Process):
    def __init__(self, args, front_sink_addr):
        super().__init__()
        self.port = args.port_out
        self.exit_flag = multiprocessing.Event()
        self.logger = set_logger(colored('SINK', 'green'), verbose=args.verbose)
        self.front_sink_addr = front_sink_addr
        self.verbose = args.verbose
        self.args = args

    def close(self):
        self.logger.info('shutting down...')
        self.exit_flag.set()
        self.terminate()
        self.join()
        self.logger.info('terminated!')

    def run(self):
        self._run()

    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @zmqd.socket(zmq.PUB)
    def _run(self, receiver, frontend, sender):
        receiver_addr = auto_bind(receiver)
        frontend.connect(self.front_sink_addr)
        sender.bind('tcp://*:%d' % self.port)

        pending_checksum = defaultdict(int)
        pending_result = defaultdict(list)
        job_checksum = defaultdict(int)

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(receiver, zmq.POLLIN)

        # send worker receiver address back to frontend
        frontend.send(receiver_addr.encode('ascii'))

        # Windows does not support logger in MP environment, thus get a new logger
        # inside the process for better compability
        logger = set_logger(colored('SINK', 'green'), verbose=self.verbose)
        logger.info('ready')

        while not self.exit_flag.is_set():
            socks = dict(poller.poll())
            if socks.get(receiver) == zmq.POLLIN:
                msg = receiver.recv_multipart()
                job_id = msg[0]
                job_info = job_id.split(b'@')
                job_id = job_info[0]
                partial_id = job_info[1] if len(job_info) == 2 else 0

                arr_info, arr_val = jsonapi.loads(msg[1]), pickle.loads(msg[2])
                pending_result[job_id].append((arr_val, partial_id))
                pending_checksum[job_id] += len(arr_val)
                logger.info('collect job\t%s (%d/%d)' % (job_id,
                                                        pending_checksum[job_id],
                                                        job_checksum[job_id]))

                # check if there are finished jobs, send it back to workers
                finished = [(k, v) for k, v in pending_result.items() if pending_checksum[k] == job_checksum[k]]
                for job_info, tmp in finished:
                    logger.info('send back\tsize: %d\tjob id:%s\t' % (job_checksum[job_info], job_info))
                    # re-sort to the original order
                    tmp = [x[0] for x in sorted(tmp, key=lambda x: int(x[1]))]
                    client_addr, req_id = job_info.split(b'#')
                    # print(tmp)
                    send_ndarray(sender, client_addr, np.concatenate(np.asarray(tmp), axis=0), req_id)
                    pending_result.pop(job_info)
                    pending_checksum.pop(job_info)
                    job_checksum.pop(job_info)

            if socks.get(frontend) == zmq.POLLIN:
                client_addr, msg_type, msg_info, req_id = frontend.recv_multipart()
                if msg_type == ServerCommand.new_job:
                    job_info = client_addr + b'#' + req_id
                    job_checksum[job_info] = int(msg_info)
                    logger.info('job register\tsize: %d\tjob id: %s' % (int(msg_info), job_info))
                elif msg_type == ServerCommand.show_config:
                    time.sleep(0.1)  # dirty fix of slow-joiner: sleep so that client receiver can connect.
                    logger.info('send config\tclient %s' % client_addr)
                    sender.send_multipart([client_addr, msg_info, req_id])


class MessageWorker(Process):
    def __init__(self, id, args, worker_address_list, sink_address, device_id):
        super().__init__()
        self.worker_id = id
        self.device_id = device_id
        self.logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), verbose=args.verbose, )
        self.daemon = True
        self.exit_flag = multiprocessing.Event()
        self.worker_address = worker_address_list
        self.num_concurrent_socket = len(self.worker_address)
        self.sink_address = sink_address
        self.prefetch_size = args.prefetch_size if self.device_id > 0 else None  # set to zero for CPU-worker
        self.gpu_memory_fraction = args.gpu_memory_fraction   ### not used
        self.verbose = args.verbose
        self.use_fp16 = args.fp16
        self.args = args

    def close(self):
        self.logger.info('shutting down...')
        self.exit_flag.set()
        self.terminate()
        self.join()
        self.logger.info('terminated!')


    def model_inference(self,contents,client_id,tf):
        '''

        :param contents:
        :param client_id:
        :param tf:
        :return:
        '''

        contents_list = contents[0]
        para_dict = contents[1]

        para_dict['tf'] = tf  # 系统分配给当前worker的 tf 其中指定了设备号

        # load model
        model_proxy = para_dict['model_proxy_path'] # 模型代理的路径，需要从MessageClient 中使用 para_dict 传入
        #print(model_proxy)
        with open(model_proxy,'rb') as pr:
            _model = pickle.load(pr)
        self.logger.info('model_parameter:\n\t\t' + '\n\t\t'.join([str(k)+': '+str(v) for k,v in para_dict.items() ]))
        # infer
        infer_rst = _model.inference(contents_list, para_dict)
        return infer_rst, client_id

    def run(self):
        self._run()

    @zmqd.socket(zmq.PUSH)
    @multi_socket(zmq.PULL, num_socket='num_concurrent_socket')
    def _run(self, sink, *receivers):
        # Windows does not support logger in MP environment, thus get a new logger
        # inside the process for better compatibility
        logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), verbose=self.verbose)

        logger.info('use device %s, for loading model and data from %s' %
                    ('cpu' if self.device_id < 0 else ('gpu: %d' % self.device_id), str('client call')))

        tf = import_tf(self.device_id, self.verbose, use_fp16=self.use_fp16)

        for sock, addr in zip(receivers, self.worker_address):
            sock.connect(addr)

        sink.connect(self.sink_address)

        for id_contents in self.input_fn_builder(receivers):
            contents = id_contents['contents']
            client_id = id_contents['client_id']

            # 判断‘model_proxy_path’和其他表示文件路径的参数 是否是绝对路径，如果是绝对路径：不处理，如果是相对路径加上 MDM服务启动时设置的‘run_script_dir’ 并判断路径是否有效
            if (not os.path.isdir(self.args.run_script_dir)) or (not os.path.exists(self.args.run_script_dir)):
                self.logger.error('the run_script_dir: %s is invalid! ' % self.args.run_script_dir)
            check_file_path_valid = True
            for k in contents[1].keys():
                if k.endswith('path'):
                    if not os.path.isabs(contents[1][k]):
                        contents[1][k] = os.path.join(self.args.run_script_dir, contents[1][k])
                        #print(contents[1]['model_proxy_path'])
                    if not os.path.exists(contents[1][k]):
                        self.logger.error('find the %s: %s is invalid! ' % (k,contents[1][k]))
                        check_file_path_valid = False
                        break
            if check_file_path_valid:
                r_label, r_c_id = self.model_inference(contents, client_id, tf)
            else:
                r_label, r_c_id = [''] * len(contents[0]), client_id
            rst = send_ndarray(sink, r_c_id, r_label)
            logger.info('job done\tsize: %s\tclient: %s' % (np.array(r_label).shape, r_c_id))

    def input_fn_builder(self, socks):
        import sys
        sys.path.append("..")

        def gen():
            logger = set_logger(colored('WORKER-%d' % self.worker_id, 'yellow'), verbose=self.verbose)
            poller = zmq.Poller()
            for sock in socks:
                poller.register(sock, zmq.POLLIN)
            logger.info('ready and listening!')

            while not self.exit_flag.is_set():
                events = dict(poller.poll())
                for sock_idx, sock in enumerate(socks):
                    if sock in events:
                        # 接收来自客户端的消息
                        client_id, raw_msg = sock.recv_multipart()
                        msg = jsonapi.loads(raw_msg)
                        # print(msg)
                        logger.info('new job\tsocket: %d\tsize: %d\tclient: %s' % (sock_idx, len(msg[0]), client_id))
                        # check if msg is a list of list, if yes consider the input is already tokenized
                        # 对接收到的字符进行切词，并且转化为id格式
                        # logger.info('get msg:%s, type:%s' % (msg[0], type(msg[0])))

                        # is_tokenized = all(isinstance(el, list) for el in msg[0])

                        # logger.info('contents: %s\nsize: %d\nmodel: %s' % (msg, len(msg[0]), self.args.mode) )
                        yield {"contents": msg,
                               "client_id": client_id}

        # return msg, client_id
        return gen()


class ServerStatistic:
    def __init__(self):
        self._hist_client = defaultdict(int)
        self._hist_msg_len = defaultdict(int)
        self._client_last_active_time = defaultdict(float)
        self._num_data_req = 0
        self._num_sys_req = 0
        self._num_total_seq = 0
        self._last_req_time = time.perf_counter()
        self._last_two_req_interval = []
        self._num_last_two_req = 200

    def update(self, request):
        client, msg, req_id, msg_len = request
        self._hist_client[client] += 1
        if ServerCommand.is_valid(msg):
            self._num_sys_req += 1
            # do not count for system request, as they are mainly for heartbeats
        else:
            self._hist_msg_len[int(msg_len)] += 1
            self._num_total_seq += int(msg_len)
            self._num_data_req += 1
            tmp = time.perf_counter()
            self._client_last_active_time[client] = tmp
            if len(self._last_two_req_interval) < self._num_last_two_req:
                self._last_two_req_interval.append(tmp - self._last_req_time)
            else:
                self._last_two_req_interval.pop(0)
            self._last_req_time = tmp

    @property
    def value(self):
        def get_min_max_avg(name, stat):
            if len(stat) > 0:
                return {
                    'avg_%s' % name: sum(stat) / len(stat),
                    'min_%s' % name: min(stat),
                    'max_%s' % name: max(stat),
                    'num_min_%s' % name: sum(v == min(stat) for v in stat),
                    'num_max_%s' % name: sum(v == max(stat) for v in stat),
                }
            else:
                return {}

        def get_num_active_client(interval=180):
            # we count a client active when its last request is within 3 min.
            now = time.perf_counter()
            return sum(1 for v in self._client_last_active_time.values() if (now - v) < interval)

        parts = [{
            'num_data_request': self._num_data_req,
            'num_total_seq': self._num_total_seq,
            'num_sys_request': self._num_sys_req,
            'num_total_request': self._num_data_req + self._num_sys_req,
            'num_total_client': len(self._hist_client),
            'num_active_client': get_num_active_client()},
            get_min_max_avg('request_per_client', self._hist_client.values()),
            get_min_max_avg('size_per_request', self._hist_msg_len.keys()),
            get_min_max_avg('last_two_interval', self._last_two_req_interval),
            get_min_max_avg('request_per_second', [1. / v for v in self._last_two_req_interval]),
        ]

        return {k: v for d in parts for k, v in d.items()}


'''
def init_predict_var(path):
    """
    初始化NER所需要的一些辅助数据
    :param path:
    :return:
    """
    label_list_file = os.path.join(path, 'label_list.pkl')
    label_list = []
    if os.path.exists(label_list_file):
        with open(label_list_file, 'rb') as fd:
            label_list = pickle.load(fd)
    num_labels = len(label_list)

    with open(os.path.join(path, 'label2id.pkl'), 'rb') as rf:
        label2id = pickle.load(rf)
        id2label = {value: key for key, value in label2id.items()}
    return num_labels, label2id, id2label
'''



