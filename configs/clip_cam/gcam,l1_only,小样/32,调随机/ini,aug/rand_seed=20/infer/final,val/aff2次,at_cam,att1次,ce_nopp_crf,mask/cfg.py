#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/3/29 16:14
@File    : cfg.py
@Software: PyCharm
@Desc    : 没有CRF情况下，寻找最优aff配置。
"""
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/aff_voc/base.py',
                      'configs/patterns/crf/crf_eval.py',
                      'configs/patterns/crf/clip_es_crf,no_pp.py', 
                      'configs/patterns/seed/save_best_·95mask.py')

cfg.rslt_dir = ...

cfg.dt.val.ini.split = 'val'
cfg.eval.seed.mask.ini.thresh = 0

# * 覆盖原配置，使其适合调参（不改变算法）。
cfg.solver.viz_cam = False  # noqa
cfg.solver.viz_score = False

# * 修改算法参数。
cfg.aff.ini.att2aff_cfg.method.n_iter = 1

cfg.aff.ini.aff_cfg.n_iters = 2
cfg.aff.ini.aff_at = 'cam'
