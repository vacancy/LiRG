# -*- coding:utf8 -*-
# File   : io.py
# Author : Jiayuan Mao
# Email  : mjy14@mails.tsinghua.edu.cn
# Date   : 2016-05-21 23:49:08
#
# This file is part of project ``Learning in Repeated Games'' 
# of course ``Game Theory''.

import numpy


def read_database(filename):
    result = list()
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            rnd = list(map(int, line.split(' ')))
            result.append(rnd)
    return numpy.array(result)


def split_database(database, split_by):
    db1 = numpy.zeros((len(database), split_by))
    db2 = numpy.zeros((len(database), (len(database[0])-split_by)))
    for i, row in enumerate(database):
        for j, col in enumerate(row):
            if j < split_by:
                db1[i][j] = col
            else:
                db2[i][j-split_by] = col
    return db1, db2


def write_database(filename, database):
    with open(filename, 'w') as f:
        for i in range(len(database)):
            for j in range(len(database[i])):
                f.write("{} ".format(int(database[i][j])))
            f.write("\n")
