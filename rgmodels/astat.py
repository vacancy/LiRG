# -*- coding:utf8 -*-
# File   : astat.py
# Author : Jiayuan Mao
# Email  : mjy14@mails.tsinghua.edu.cn
# Date   : 2016-05-22 11:11:39
#
# This file is part of project ``Learning in Repeated Games'' 
# of course ``Game Theory''.


from liblirg import *


class Model(ModelBase):
    """
        Model record the opponent's behaviour
        The agent always choose the action that beats the opponent
    """
    def __init__(self):
        self.stat = None

    def train(self, in_data, nr_rows, nr_cols, nr_types):
        self.stat = numpy.zeros((nr_rows, nr_types), dtype=numpy.float32)
        for i in range(nr_rows):
            for j in range(nr_cols):
                self.stat[i, in_data[i][j]] += 1.0


    def predict(self, out_data, nr_rows, nr_cols, nr_types, gt_data):
        for i in range(0, nr_rows//2, 1):
            u1, u2 = 2*i, 2*i+1
            for j in range(nr_cols):
                out_data[u1, j] = (utils.argmax(self.stat[u2]) + 2) % nr_types
                out_data[u2, j] = (utils.argmax(self.stat[u1]) + 2) % nr_types
                self.stat[u1, gt_data[u1, j]] += 1
                self.stat[u2, gt_data[u2, j]] += 1
        # print(out_data)
