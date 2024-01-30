#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/16 16:57
@File    : resume_cam_format.py
@Software: PyCharm
@Desc    : 
"""
import os
import os.path as osp
import argparse

import numpy as np
from tqdm import tqdm

from libs.data import VOCAug2
from libs.seeding.score import cam2score

# * 接收参数，确定输入输出路径。
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str)
parser.add_argument('-o', '--output', type=str)
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

dt = VOCAug2('datasets', split='train_aug')

# * 列出输入路径下的所有文件。
cam_names = os.listdir(args.input)

for name in tqdm(cam_names, dynamic_ncols=True):
    img_id = osp.splitext(name)[0]
    dsize = dt.get_by_img_id(img_id).img.shape[:2]

    load = np.load(osp.join(args.input, name))

    highres = cam2score(load['cam'].astype(np.float32), dsize, resize_first=False)
    keys = load['fg_cls']

    np.save(os.path.join(args.output, img_id + '.npy'),
            {"keys": keys,
             "highres": highres,
             })
