#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/4/13 23:19
@File    : base.py
@Software: PyCharm
@Desc    : 
"""
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/clip_cam/离线伪真,CI/l1/base.py')

# * 使用cl_loss，5100 val时的伪真值作为监督。
cfg.dt.train.ini.ps_mask_dir = 'experiment/clip_cam/调cls/cl_loss/infer/5100/aff2次,at_score,ce_npp_crf,' \
                               'mask/seed/best/mask'
