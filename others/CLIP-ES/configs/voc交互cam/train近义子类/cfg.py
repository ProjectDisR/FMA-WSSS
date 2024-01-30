#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/2/16 15:17
@File    : voc交互cam.py
@Software: PyCharm
@Desc    : 
"""
import numpy as np

from alchemy_cat.py_tools import Config
from alchemy_cat.acplot import square


cfg = config = Config(cfgs_update_at_parser=('others/CLIP-ES/configs/voc交互cam/base.py',))

cfg.rslt_dir = ...

# * 15张含有火车图片。
cfg.img_names = [(s_name := str(name))[:4] + '_' + s_name[4:]
                 for name, label in np.load('./datasets/voc_cls_labels.npy', allow_pickle=True).tolist().items()
                 if label[18] == 1][:15]

# * 映射火车类别。
cfg.ori2new_class_names_changed = {'train': ('train', 'high-speed train',
                                             'freight car', 'electric locomotive', 'steam locomotive'),
                                   'bird': ('bird avian',),
                                   'chair': ('chair seat',),
                                   'person': ('person with clothes,people,human',),
                                   'tvmonitor': ('tvmonitor screen',)
                                   }

# * 指定前景间是否独立（而不是互斥）。
cfg.indep_fg = True

# * 指定作图格式。
cfg.show_row_col = square
