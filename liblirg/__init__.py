# -*- coding:utf8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : mjy14@mails.tsinghua.edu.cn
# Date   : 2016-05-22 00:05:19
#
# This file is part of project ``Learning in Repeated Games'' 
# of course ``Game Theory''.


class _G(dict):
    def __getattr__(self, k):
        return self[k]

    def __setattr__(self, k, v):
        self[k] = v

    def __delattr__(self, k):
        del self[k]

g = _G()

import __config__
from . import io, utils
from .model import ModelBase
import numpy
import numpy as np
