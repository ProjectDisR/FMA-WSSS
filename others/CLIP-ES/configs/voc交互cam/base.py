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
from itertools import chain

from alchemy_cat.py_tools import Config, ItemLazy


cfg = config = Config()

# * 设置图片名。
cfg.img_names = [name.strip() for name in tuple(open('others/CLIP-ES/voc12/train_aug.txt', 'r'))[:20]]

# * 设置类别名。
# ** 背景名。
cfg.bg_names = ['ground', 'land', 'grass', 'tree', 'building',
                'wall', 'sky', 'lake', 'water', 'river', 'sea',
                'railway', 'railroad', 'keyboard', 'helmet', 'cloud',
                'house', 'mountain', 'ocean', 'road', 'rock',
                'street', 'valley', 'bridge', 'sign',
                ]
# ** 数据集前景名。
cfg.ori_class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                       'bus', 'car', 'cat', 'chair', 'cow',
                       'diningtable', 'dog', 'horse', 'motorbike', 'person',
                       'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
                       ]
# ** 数据集转prompt前景名。
cfg.ori2new_class_names_changed = {'bird': ('bird avian',),
                                   'chair': ('chair seat',),
                                   'person': ('person with clothes,people,human',),
                                   'tvmonitor': ('tvmonitor screen',)
                                   }
cfg.ori2new_class_names = ItemLazy(lambda c: {class_name: c.ori2new_class_names_changed.get(class_name, (class_name,))
                                              for class_name in c.ori_class_names}, priority=-1)
# ** prompt前景名。
cfg.new_class_names = ItemLazy(lambda c: list(chain(*c.ori2new_class_names.values())), priority=0)
# ** prompt前景名转背景名。
def get_new2ori_class_names(c):  # noqa
    new2ori_class_names = {}
    for ori_name, new_names in c.ori2new_class_names.items():
        new2ori_class_names.update({new_name: ori_name for new_name in new_names})
    return new2ori_class_names
cfg.new2ori_class_names = ItemLazy(get_new2ori_class_names)  # noqa

# * 设置模板。
cfg.templates = ['a clean origami {}.']

# * 设置增强。
cfg.scales = [1.0]
cfg.flip = False
