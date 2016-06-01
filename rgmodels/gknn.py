# -*- coding:utf8 -*-
# File   : gsvm.py
# Author : Jiayuan Mao
# Email  : mjy14@mails.tsinghua.edu.cn
# Date   : 2016-05-22 11:11:39
#
# This file is part of project ``Learning in Repeated Games'' 
# of course ``Game Theory''.


from liblirg import *
from sklearn import svm, neighbors

g.nr_queue_size = 2
g.svm_degree = 3
g.svm_kernel = 'poly'
g.nr_neighbours = 10


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
        return result


class KNN(object):
    def __init__(self, nr_neighbours):
        self.nr_neighbours = nr_neighbours
        self.kdtree = None
        self.label = None

    def fit(self, data, label):
        self.kdtree = neighbors.NearestNeighbors(n_neighbors=self.nr_neighbours, algorithm='ball_tree').fit(data)
        self.label = label

    def predict(self, data):
        dis, index = self.kdtree.kneighbors(data)
        dis, index = dis[0], index[0]

        cnt = [0, 0, 0]
        for i in range(len(index)):
            cnt[self.label[index[i]]] += 1# / (dis[i]**2 + 0.01)
        return [utils.argmax(cnt)]

class Model(ModelBase):
    """
    Always output the action that the given agent have done most frequently.
    """
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
                    data.append(state.get_tuple())
                    label.append(in_data[i][j])
                state.push(in_data[i][j], in_data[i^1][j])
            # print(len(data), len(label))
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

                out_data[u1, j] = m1.predict([s1.get_tuple()])[0]
                out_data[u2, j] = m2.predict([s2.get_tuple()])[0]
                s1.push(gt_data[u1, j], gt_data[u2, j])
                s2.push(gt_data[u2, j], gt_data[u1, j])

                data1.append(s1.get_tuple())
                label1.append(gt_data[u1][j])
                data2.append(s2.get_tuple())
                label2.append(gt_data[u2][j])
