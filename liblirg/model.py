# -*- coding:utf8 -*-
# File   : model.py
# Author : Jiayuan Mao
# Email  : mjy14@mails.tsinghua.edu.cn
# Date   : 2016-05-22 10:28:23
#
# This file is part of project ``Learning in Repeated Games'' 
# of course ``Game Theory''.


class ModelBase(object):
    def __init__(self):
        pass

    def train(self, in_data, nr_rows, nr_cols, nr_types):
        raise NotImplementedError()

    def predict(self, out_data, nr_rows, nr_cols, nr_types):
        raise NotImplementedError()
