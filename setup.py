# -*- coding: utf-8 -*-
# @Time    : 2019/5/20 16:32
# @Author  : Mat
# @Email   : 18166034717@163.com
# @File    : setup.py
# @Software: PyCharm

from setuptools import setup,find_packages

setup(
    name = 'messageDisMiddle',
    version = '0.0.1',
    keywords = ('pip','message queue','hua-cloud'),
    description = 'message distribution middleware sdk',
    long_description = 'message distribution middleware sdk for any machine learning alg model',
    license = "MIT Licence",

    url = 'http://www.hua-cloud.com',
    author = 'zhaojq',
    author_email = 'zhaoxk1992@gmail.com',

    packages = find_packages(),
    #include_package_data = True,
    platforms = 'any',
    install_requires = ['numpy','six','pyzmq>=17.1.0','GPUtil>=1.3.0','termcolor>=1.1'],
    extras_require={
        'cpu': ['tensorflow>=1.10.0'],
        'gpu': ['tensorflow-gpu>=1.10.0'],
        'http': ['flask', 'flask-compress', 'flask-cors', 'flask-json', 'bert-serving-client']
     },
    entry_points={
        'console_scripts':['MDM-server-start=messageDisMiddle.run_server:start_server',
                           'MDM-server-terminate=messageDisMiddle.stop_server:stop_server']
    }
)