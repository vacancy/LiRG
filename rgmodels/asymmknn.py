# -*- coding:utf8 -*-
# File   : asymmsvm.py
# Author : Jiayuan Mao
# Email  : mjy14@mails.tsinghua.edu.cn
# Date   : 2016-05-22 11:11:39
#
# This file is part of project ``Learning in Repeated Games'' 
# of course ``Game Theory''.


from liblirg import *
from sklearn import svm
from rgmodels.gknn import KNN

g.nr_queue_size = 2
g.nr_neighbours = 3
g.svm_degree = 10

class State(object):
    def __init__(self, max_len):
        self.mine = list()
        self.oppo = list()
        self.max_len = max_len

    def push(self, v1, v2):
        self.push_mine(v1)
        self.push_oppo(v2)
    def push_mine(self, v):
        self.mine.append(v)
        if (len(self.mine) > self.max_len):
            self.mine = self.mine[1:]
    def push_oppo(self, v):
        self.oppo.append(v)
        if (len(self.oppo) > self.max_len):
            self.oppo = self.oppo[1:]
    def ok(self):
        return len(self.mine) == len(self.oppo) == self.max_len
    def get_tuple(self):
        result = list()
        result.extend(self.mine)
        result.extend(self.oppo)
        a, result = result[0], result[1:]
        result = list(map(lambda x: (x-a+3)%3, result))
        return a, result


class Model(ModelBase):
    def __init__(self):
        self.states = list()
        self.datas = list()
        self.labels = list()

    def train(self, in_data, nr_rows, nr_cols, nr_types):
        self.stat = numpy.zeros((nr_rows, nr_types), dtype=numpy.float32)
        for i in range(nr_rows):
            data = list()
            label = list()
            state = State(g.nr_queue_size)
            for j in range(nr_cols):
                if state.ok():
                    a, result = state.get_tuple()
                    data.append(result)
                    label.append((in_data[i][j]-a+3)%3)
                state.push(in_data[i][j], in_data[i^1][j])

            self.states.append(state)
            self.datas.append(data)
            self.labels.append(label)


    def predict(self, out_data, nr_rows, nr_cols, nr_types, gt_data):
        for i in range(0, nr_rows//2, 1):
            u1, u2 = 2*i, 2*i+1
            s1, s2 = self.states[u1:u2+1]
            data1, data2 = self.datas[u1:u2+1]
            label1, label2 = self.labels[u1:u2+1]
            for j in range(nr_cols):
                m1 = KNN(g.nr_neighbours)
                m1.fit(data1, label1)
                m2 = KNN(g.nr_neighbours)
                m2.fit(data2, label2)
                
                a, result = s1.get_tuple()
                out_data[u1, j] = (m1.predict([result])[0] + a) % 3
                a, result = s2.get_tuple()
                out_data[u2, j] = (m2.predict([result])[0] + a) % 3

                s1.push(gt_data[u1, j], gt_data[u2, j])
                a, result = s1.get_tuple()
                data1.append(result)
                label1.append((gt_data[u1][j]-a+3)%3)

                s2.push(gt_data[u2, j], gt_data[u1, j])
                a, result = s2.get_tuple()
                data2.append(result)
                label2.append((gt_data[u2][j]-a+3)%3)
