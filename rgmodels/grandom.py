# -*- coding:utf8 -*-
# File   : grandom.py
# Author : Jiayuan Mao
# Email  : mjy14@mails.tsinghua.edu.cn
# Date   : 2016-05-22 10:28:23
#
# This file is part of project ``Learning in Repeated Games'' 
# of course ``Game Theory''.


from liblirg import *


class Model(ModelBase):
    def train(self, in_data, nr_rows, nr_cols, nr_types):
        pass

    def predict(self, out_data, nr_rows, nr_cols, nr_types, gt_data):
        for i in range(nr_rows):
            for j in range(nr_cols):
                out_data[i][j] = utils.randint(nr_types)
