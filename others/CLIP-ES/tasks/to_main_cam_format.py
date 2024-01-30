#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/16 16:23
@File    : convert-cam-format.py
@Software: PyCharm
@Desc    : 
"""
import os
import os.path as osp

import numpy as np
import argparse

# * 接收参数，确定输入输出路径。
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str)
parser.add_argument('-o', '--output', type=str)
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

# * 列出输入路径下的所有文件。
score_names = os.listdir(args.input)

for name in score_names:
    load = np.load(osp.join(args.input, name), allow_pickle=True).item()
    score = load['highres'].astype(np.float32)
    fg_cls = load['keys']

    np.savez(osp.join(args.output, f'{osp.splitext(name)[0]}.npz'), cam=score, fg_cls=fg_cls)
