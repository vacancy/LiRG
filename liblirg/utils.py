# -*- coding:utf8 -*-
# File   : utils.py
# Author : Jiayuan Mao
# Email  : mjy14@mails.tsinghua.edu.cn
# Date   : 2016-05-21 23:49:08
#
# This file is part of project ``Learning in Repeated Games'' 
# of course ``Game Theory''.


import numpy
import numpy.random


def randint(nr_type):
    return numpy.random.randint(0, nr_type)


def randprob(probs):
    r = numpy.random.rand()
    accumulate = 0
    for i, p in enumerate(probs):
        accumulate += p
        if r < p:
            return i
    return len(probs) - 1

def argmax(probs):
	return numpy.argmax(probs)
