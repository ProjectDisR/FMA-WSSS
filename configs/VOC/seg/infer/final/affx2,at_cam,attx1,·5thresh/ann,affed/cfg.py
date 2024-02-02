#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/5/25 16:05
@File    : cfg.py
@Software: PyCharm
@Desc    :
"""
from alchemy_cat.py_tools import Config

cfg = config = Config('configs/anns_seed/base.py')

cfg.rslt_dir = ...

cfg.dt.ini.split = 'train'

# * 配置CAM路径。
cfg.cam.dir = 'experiment/VOC/seg/infer/final/affx2,at_cam,attx1,·5thresh/cam_affed'

# * 配置替补种子点路径。
cfg.seed.dir = None

# * 配置SAM标注路径。
cfg.sam_anns.dir = 'experiment/VOC/sams/split=train_aug/anns'

# * 修改得分算法参数。
cfg.seed.norm_firsts = [True]
cfg.seed.bg_methods = [{'method': 'pow', 'pow': i} for i in [0.5, 1.0, 1.5, 2.0, 3.0]]