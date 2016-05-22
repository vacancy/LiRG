# -*- coding:utf8 -*-
# File   : train.py
# Author : Jiayuan Mao
# Email  : mjy14@mails.tsinghua.edu.cn
# Date   : 2016-05-22 00:02:48
#
# This file is part of project ``Learning in Repeated Games'' 
# of course ``Game Theory''.


from liblirg import *
from evaluate import evaluate

import argparse
import importlib
import sys
import numpy


def load_model(name):
    sys.path.insert(0, 'rgmodels')
    if name.endswith('.py'):
        name = name[0:-3]
    if name.startswith('rgmodels/'):
        name = name[len('rgmodels/'):]
    desc = importlib.import_module(name)
    del sys.path[0]
    return desc.Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', help='Training data file, e.g. data/1_train.txt')
    parser.add_argument('-r', '--oround', dest='oround', default=20, help='Number of output rounds, e.g. 20 [default=20]')
    parser.add_argument('-m', '--model', dest='model', help='Model python library, e.g. modelxxx, should located in \'rgmodels\'')

    parser.add_argument('-o', '--ofile', dest='ofile', help='Output data file, e.g. result/1_modelxxx.txt')
    parser.add_argument('-gt', '--gt', dest='groundtruth', help='Groundtruth, use to test the model if provided')
    args = parser.parse_args()

    assert args.input is not None
    assert args.model is not None

    in_data = io.read_database(args.input)
    nr_rows = len(in_data)
    nr_cols = len(in_data[0])
    nr_types = numpy.array(in_data).max()+1
    out_data = numpy.zeros((nr_rows, args.oround), dtype=numpy.int32)

    model = load_model(args.model)()
    model.train(in_data, nr_rows, nr_cols, nr_types)
    model.predict(out_data, nr_rows, args.oround, nr_types)

    if args.ofile is not None:
        io.write_database(args.ofile)
    if args.groundtruth is not None:
        gt_data = io.read_database(args.groundtruth)
        score, count = evaluate(out_data, gt_data)
        print("model={} score={}/{}".format(args.model, score, count))


if __name__ == '__main__':
    main()
