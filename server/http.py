from multiprocessing import Process

from termcolor import colored

from .helper import set_logger,str2bool

import json


class HTTPProxy(Process):
    def __init__(self, args):
        super().__init__()
        self.args = args


    def create_flask_app(self):
        try:
            from flask import Flask, request
            from flask_compress import Compress
            from flask_cors import CORS
            from flask_json import FlaskJSON, as_json, JsonError
            from client import ConcurrentMessageClient
        except ImportError:
            raise ImportError('Client or Flask or its dependencies are not fully installed, '
                              'they are required for serving HTTP requests.'
                              'Please install it.')

        # support up to 10 concurrent HTTP requests
        cmc = ConcurrentMessageClient(max_concurrency=self.args.http_max_connect,
                                  port=self.args.port, port_out=self.args.port_out,
                                  output_fmt='list')
        app = Flask(__name__)
        logger = set_logger(colored('PROXY', 'red'), logfile=self.args.logger)

        # 服务端状态
        @app.route('/status/server', methods=['GET'])
        @as_json
        def get_server_status():
            logger.info( 'http request (/status/server) from %s' % request.remote_addr)
            return cmc.server_status

        # 客户端状态
        @app.route('/status/client', methods=['GET'])
        @as_json
        def get_client_status():
            logger.info('http request (/status/client) from %s' % request.remote_addr)
            return cmc.status

        # 识别
        @app.route('/ner', methods=['POST'])
        @as_json
        def encode_query():
            data = request.form if request.form else request.json
            try:
                logger.info('http request (/ner %s) from %s' % ('',request.remote_addr))
                logger.debug('http request (/ner %s) from %s' % (data, request.remote_addr))
                #print(request.remote_addr)
                #print(str2bool(data['tag_independent']) if 'tag_independent' in data else False,data.getlist("texts"),type(data.getlist("texts")) )
                #print('sdsdfsdfsd',data['tag_independent'])
                # 字符串
                texts_dict = data["texts"]
                #texts_dict = json.loads(data["texts"])
                #texts_dict = json.loads(json.dumps(data["texts"]) )
                #print(texts_dict,type(texts_dict))
                texts_list = [texts_dict[str(key)] for key in sorted([int(i) for i in texts_dict.keys()])]

                # 字符串idx
                idxs_dict = data['idxs']
                #idxs_dict = json.loads(json.dumps(data['idxs']))
                idxs_list = [idxs_dict[str(key)] for key in sorted([int(i) for i in idxs_dict.keys()])]

                # 校验
                if len(texts_list) == len(idxs_list):
                    idxs_sp_list = []
                    for i, term in enumerate(texts_list):
                        idxs_sp_list.append(idxs_list[i].split(','))
                        if len(term) != len( idxs_list[i].split(',') ):
                            logger.error('http request texts seqs len and idxs seqs len unequal in seqs %s.' % str(i))
                            return {'id': data['id'],
                                    'tag': data['tag'],
                                    'texts': [],
                                    'idxs': []
                                    }
                else:
                    logger.error('http request texts sum and idxssum unequal.')
                    return {'id': data['id'],
                            'tag': data['tag'],
                            'texts': [],
                            'idxs': []
                            }

                # 调用推理函数
                inf_res = cmc.message_handle(
                    texts_list,
                    data['tag'],
                    tag_independent=str2bool(data['tag_independent']) if 'tag_independent' in data else False,
                    is_tokenized=str2bool(data['is_tokenized']) if 'is_tokenized' in data else False
                )
                # 溯源
                tag_idxs = []
                for i,tag_list in enumerate(inf_res):
                    idxs = []
                    for tag_one in tag_list:
                        s_tag_idx = texts_list[i].find(tag_one)
                        #print(tag_one)
                        #print(texts_list[i])
                        #print(idxs_sp_list[i])
                        #print('s_tag_idx',s_tag_idx)
                        e_tag_idx = s_tag_idx + len(tag_one)
                        #print('e_tag_idx',e_tag_idx)
                        #print(','.join(idxs_sp_list[i][s_tag_idx:e_tag_idx]))
                        idxs.append(','.join(idxs_sp_list[i][s_tag_idx:e_tag_idx]))
                        
                    tag_idxs.append(idxs)
                #print(data['texts'])
                #texts_list = []
                #texts_list.append(data['texts']) # 将一个序列放进list中.
                return {'id': data['id'],
                        'tag': data['tag'],
                        'texts': inf_res,
                        'idxs': tag_idxs
                        }

            except Exception as e:
                logger.error('http request error (/ner %s) from %s' % ('',request.remote_addr), exc_info=True)
                raise JsonError(description=str(e), type=str(type(e).__name__))

        CORS(app, origins=self.args.cors)
        FlaskJSON(app)
        Compress().init_app(app)
        return app

    def run(self):
        app = self.create_flask_app()

        #from werkzeug.contrib.fixers import ProxyFix
        #app.wsgi_app = ProxyFix(app.wsgi_app)

        app.run(port=self.args.http_port, threaded=True, host='0.0.0.0')
