# -*- coding:utf8 -*-
# File   : gmc2.py
# Author : Jiayuan Mao
# Email  : mjy14@mails.tsinghua.edu.cn
# Date   : 2016-05-22 11:11:39
#
# This file is part of project ``Learning in Repeated Games'' 
# of course ``Game Theory''.


from liblirg import *


class Model(ModelBase):
    """
    Markov Chain v2:
        input: last action, last opponent's action
        output: this action 
    """
    def __init__(self):
        self.mc = None
        self.in_data = None

    def train(self, in_data, nr_rows, nr_cols, nr_types):
        self.mc = numpy.ones((nr_rows, nr_types, nr_types, nr_types), dtype=numpy.float32)
        self.in_data = in_data = numpy.array(in_data)
        for i in range(nr_rows):
            for j in range(1, nr_cols):
                self.mc[i][in_data[i, j-1], in_data[i^1, j-1], in_data[i, j]] += 1

    def predict(self, out_data, nr_rows, nr_cols, nr_types, gt_data):
        for i in range(0, nr_rows//2, 1):
            last1 = self.in_data[2*i, -1]
            last2 = self.in_data[2*i+1, -1]
            u1, u2 = 2*i, 2*i+1
            for j in range(nr_cols):
                out_data[u1, j] = utils.argmax(self.mc[u1][last1, last2])
                out_data[u2, j] = utils.argmax(self.mc[u2][last2, last1])
                self.mc[i][last1, last2, gt_data[u1, j]] += 1
                self.mc[i][last2, last1, gt_data[u2, j]] += 1
                last1 = gt_data[u1, j]
                last2 = gt_data[u2, j]

