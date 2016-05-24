# -*- coding:utf8 -*-
# File   : asimple.py
# Author : Jiayuan Mao
# Email  : mjy14@mails.tsinghua.edu.cn
# Date   : 2016-05-24 12:37:59
#
# This file is part of project ``Learning in Repeated Games'' 
# of course ``Game Theory''.


from liblirg import *


class Model(ModelBase):
    def __init__(self):
        self.in_data = None

    def train(self, in_data, nr_rows, nr_cols, nr_types):
        self.in_data = in_data = numpy.array(in_data)

    def predict(self, out_data, nr_rows, nr_cols, nr_types, gt_data):
        for i in range(nr_rows):
            last = self.in_data[i^1, -1]
            for j in range(nr_cols):
                out_data[i, j] = (last + 1) % 3
                last = gt_data[i^1, j]
