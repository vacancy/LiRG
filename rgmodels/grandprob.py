# -*- coding:utf8 -*-
# File   : grandprob.py
# Author : Jiayuan Mao
# Email  : mjy14@mails.tsinghua.edu.cn
# Date   : 2016-05-22 11:11:39
#
# This file is part of project ``Learning in Repeated Games'' 
# of course ``Game Theory''.


from liblirg import *


class Model(ModelBase):
    def __init__(self):
        self.stat = None

    def train(self, in_data, nr_rows, nr_cols, nr_types):
        self.stat = numpy.zeros((nr_rows, nr_types), dtype=numpy.float32)
        for i in range(nr_rows):
            for j in range(nr_cols):
                self.stat[i, in_data[i][j]] += 1.0 / nr_cols


    def predict(self, out_data, nr_rows, nr_cols, nr_types, gt_data):
        for i in range(nr_rows):
            for j in range(nr_cols):
                out_data[i][j] = utils.randprob(self.stat[i])
