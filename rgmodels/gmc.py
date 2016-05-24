# -*- coding:utf8 -*-
# File   : gmc.py
# Author : Jiayuan Mao
# Email  : mjy14@mails.tsinghua.edu.cn
# Date   : 2016-05-22 11:11:39
#
# This file is part of project ``Learning in Repeated Games'' 
# of course ``Game Theory''.


from liblirg import *


class Model(ModelBase):
    """
    Markov Chain:
        input: last action
        output: this action 
    """   
    def __init__(self):
        self.mc = None
        self.in_data = None

    def _norm(self, x, default_):
        if x.sum() > 0:
            return x / x.sum()
        else:
            return numpy.array([default_ for _ in range(len(x))])

    def train(self, in_data, nr_rows, nr_cols, nr_types):
        self.mc = numpy.zeros((nr_rows, nr_types, nr_types), dtype=numpy.float32)
        self.in_data = in_data = numpy.array(in_data)
        for i in range(nr_rows):
            for j in range(1, nr_cols):
                self.mc[i][in_data[i, j-1], in_data[i, j]] += 1

    def predict(self, out_data, nr_rows, nr_cols, nr_types, gt_data):
        for i in range(nr_rows):
            last = self.in_data[i, -1]
            for j in range(nr_cols):
                prob = self._norm(self.mc[i][last], 1 / nr_types)
                out_data[i, j] = utils.randprob(prob)
                self.mc[i][last, gt_data[i, j]] += 1
                last = gt_data[i, j]
