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
        # for out in outs:
        #     assert len(out) == len(gt), "Different result length nr_pred={}, nr_gt={}".format(len(out), len(gt))
        #     for i in range(len(gt)):
        #         assert len(out[i]) == len(gt[i]), "Different result length in line {} nr_pred={}, nr_gt={}".format(i, len(out[i]), len(gt[i]))

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
    parser.add_argument('-val', '--val', dest='val_split', type=int, default=15, help='Split input data to train, val.')

    parser.add_argument('-o', '--ofile', dest='ofile', help='Output data file, e.g. result/1_modelxxx.txt')
    parser.add_argument('-gt', '--gt', dest='groundtruth', help='Groundtruth, use to test the model if provided')
    args = parser.parse_args()

    assert args.input is not None
    assert args.model is not None

    in_data = io.read_database(args.input)
    in_data = numpy.array(in_data)

    assert args.groundtruth is not None
    gt_data = io.read_database(args.groundtruth)

    train_data, val_data = in_data[:, 0:args.val_split], in_data[:, args.val_split:]

    nr_rows = len(train_data)
    nr_cols = len(in_data[0])
    nr_train_cols = len(train_data[0])
    nr_val_cols = len(val_data[0])
    nr_types = numpy.array(train_data).max()+1

    all_outs = list()
    for current_model in args.model:
        current_outs = list()
        for _ in range(10):
            out_data = numpy.zeros((nr_rows, nr_val_cols), dtype=numpy.int32)
            model = load_model(current_model)()
            model.train(train_data, nr_rows, nr_train_cols, nr_types)
            model.predict(out_data, nr_rows, nr_val_cols, nr_types, val_data)
            current_outs.append(out_data)
        all_outs.append(current_outs)
    final_map = ensemble(all_outs, gt=val_data)
    # print(final_map)

    all_outs = list()
    for current_model in args.model:
        out_data = numpy.zeros((nr_rows, args.oround), dtype=numpy.int32)
        model = load_model(current_model)()
        model.train(in_data, nr_rows, nr_cols, nr_types)
        model.predict(out_data, nr_rows, args.oround, nr_types, gt_data)
        all_outs.append(out_data)
    final = ensemble(all_outs, final_map=final_map)
    # final = cheat_ensemble(all_outs, gt_data)

    score, count = evaluate(final, gt_data)
    print("model={} score={}/{}".format(args.model, score, count))


if __name__ == '__main__':
    main()
