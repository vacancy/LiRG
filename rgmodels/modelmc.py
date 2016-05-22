# -*- coding:utf8 -*-
# File   : modelmc.py
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
    mc = None
    in_data = None

    def train(self, in_data, nr_rows, nr_cols, nr_types):
        self.mc = numpy.zeros((nr_rows, nr_types, nr_types), dtype=numpy.float32)
        self.in_data = in_data = numpy.array(in_data)
        for i in range(nr_rows):
            for j in range(1, nr_cols):
                self.mc[i][in_data[i, j-1], in_data[i, j]] ++ 1
        for i in range(nr_rows):
            for k in range(nr_types):
                self.mc[i, k] /= self.mc[i, k].sum() + 1e-8


    def predict(self, out_data, nr_rows, nr_cols, nr_types):
        for i in range(nr_rows):
            last = self.in_data[i, -1]
            for j in range(nr_cols):
                last = out_data[i, j] = utils.randprob(self.mc[i][last])
