# -*- coding: utf-8 -*-
# @Time    : 2019/5/16 9:11
# @Author  : Mat
# @Email   : 18166034717@163.com
# @File    : model.py
# @Software: PyCharm

import os
import time
import random
import pickle
from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self,model_name):
        self.model_name = model_name

    @abstractmethod
    def dump(self,**kwargs):
        pass

    # @abstractmethod
    # def load(self,model_proxy_path):
    #     pass

    @abstractmethod
    def inference(self,model_proxy_path):
        pass


class testModel(Model):
    pass


if __name__ == '__main__':

    testModel()
    '''
    该基类定义了dump() 和 inference() 两个抽象方法。
    重写dump()后，使用dump方法生成模型代理文件，消息中间件对通过load该模型代理文件可加载模型代理。
    模型代理加载完之后，会生成模型对象，消息中间件会调用模型对象中重写的inference方法分布式执行推理，返回结果。
    注意：所有模型推理的python脚本必须继承该抽象基类Model，并且需要强制重写这两个抽象方法，否则报错。
    
    Traceback (most recent call last):
      File "model.py", line 31, in <module>
        testModel()
    TypeError: Can't instantiate abstract class testModel with abstract methods __init__, dump, inference
    '''

