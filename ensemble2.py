# -*- coding:utf8 -*-
# File   : ensemble.py
# Author : Jiayuan Mao
# Email  : mjy14@mails.tsinghua.edu.cn
# Date   : 2016-05-22 00:02:48
#
# This file is part of project ``Learning in Repeated Games'' 
# of course ``Game Theory''.


from liblirg import *
from evaluate import evaluate
from train import load_model

import argparse
import importlib
import sys
import numpy


def ensemble(outs, gt=None, final_map=None):
    if gt is not None and final_map is None:
        final_map = numpy.zeros((len(gt), ), dtype=numpy.int32)

        for i in range(len(gt)):
            best_score = 0
            for out_id, out in enumerate(outs):
                score = 0
                for sample in out:
                    for j in range(len(sample[i])):
                        score += 1 if sample[i][j] == gt[i][j] else 0
                if score > best_score:
                    best_score = score
                    final_map[i] = out_id
        return final_map
    else:
        assert final_map is not None
        final = numpy.zeros_like(outs[0])
        for i in range(len(outs[0])):
            final[i, :] = outs[final_map[i]][i, :]
        return final


def cheat_ensemble(outs, gt):
    for out in outs:
        assert len(out) == len(gt), "Different result length nr_pred={}, nr_gt={}".format(len(out), len(gt))
        for i in range(len(gt)):
            assert len(out[i]) == len(gt[i]), "Different result length in line {} nr_pred={}, nr_gt={}".format(i, len(out[i]), len(gt[i]))

    final = numpy.zeros_like(gt)

    for i in range(len(gt)):
        best_score = 0
        best_id = -1
        for out_id, out in enumerate(outs):
            score = 0
            for j in range(len(out[i])):
                score += 1 if out[i][j] == gt[i][j] else 0
            if score > best_score:
                best_score = score
                final[i, :] = out[i, :]
                best_id = out_id
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='input', help='Training data file, e.g. data/1_train.txt')
    parser.add_argument('-r', '--oround', dest='oround', default=20, help='Number of output rounds, e.g. 20 [default=20]')
    parser.add_argument('-m', '--model', dest='model', nargs='+', help='Model python library, e.g. modelxxx, should located in \'rgmodels\'')

    parser.add_argument('-o', '--ofile', dest='ofile', help='Output data file, e.g. result/1_modelxxx.txt')
    parser.add_argument('-gt', '--gt', dest='groundtruth', help='Groundtruth, use to test the model if provided')
    args = parser.parse_args()

    assert args.input is not None
    assert args.model is not None
    assert args.groundtruth is not None

    in_data = io.read_database(args.input)
    in_data = numpy.array(in_data)
    gt_data = io.read_database(args.groundtruth)
    gt_data = numpy.array(gt_data)
    all_data = numpy.concatenate([in_data, gt_data], axis=1)

    nr_rows = len(all_data)
    nr_types = numpy.array(all_data).max()+1

    final_data = numpy.zeros_like(gt_data)
    for i in range(30, 50):
        for j1 in range(all_data.shape[0]):
            best_score, best_model = -1, None
            for current_model in args.model:
                sum_score = 0
                for _ in range(1):
                    train_data = all_data[:, 0:25]
                    val_data = all_data[:, 25:i]
                    out_data = numpy.zeros_like(val_data)
                    model = load_model(current_model)()
                    model.train(train_data, nr_rows, train_data.shape[1], nr_types)
                    model.predict(out_data, nr_rows, val_data.shape[1], nr_types, val_data)
                    for j2 in range(val_data.shape[1]):
                        sum_score += 1 if val_data[j1, j2] == out_data[j1, j2] else 0
                if sum_score > best_score:
                    best_score = sum_score
                    best_model = current_model

            current_final_out = numpy.zeros_like(all_data[:, i:i+1])
            print(i, j1, best_model)
            model = load_model(best_model)()
            model.train(all_data[:, 0:i], nr_rows, i, nr_types)
            model.predict(current_final_out, nr_rows, 1, nr_types, all_data[:, i:i+1])
            final_data[j1, i-30] = current_final_out[j1, 0]

    score, count = evaluate(final_data, gt_data)
    print("model={} score={}/{}".format(args.model, score, count))


if __name__ == '__main__':
    main()
