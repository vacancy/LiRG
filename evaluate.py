# -*- coding:utf8 -*-
# File   : evaluate.py
# Author : Jiayuan Mao
# Email  : mjy14@mails.tsinghua.edu.cn
# Date   : 2016-05-22 00:02:48
#
# This file is part of project ``Learning in Repeated Games'' 
# of course ``Game Theory''.


from liblirg import *

import argparse

def evaluate(pred, gt):
    score, count = 0, 0
    assert len(pred) == len(gt), "Different result length nr_pred={}, nr_gt={}".format(len(pred), len(gt))
    for i in range(len(pred)):
        assert len(pred[i]) == len(gt[i]), "Different result length in line {} nr_pred={}, nr_gt={}".format(i, len(pred[i]), len(gt[i]))
        for j in range(len(pred[i])):
            score += 1 if pred[i][j] == gt[i][j] else 0
            count += 1
    return score, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred')
    parser.add_argument('gt')
    args = parser.parse_args()

    a = io.read_database(args.pred)
    b = io.read_database(args.gt)
    score, count = evaluate(a, b)
    print("score={}/{}".format(score, count))


if __name__ == '__main__':
    main()
