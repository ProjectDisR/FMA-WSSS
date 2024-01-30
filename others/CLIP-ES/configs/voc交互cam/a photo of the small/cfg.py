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
from alchemy_cat.py_tools import Config

cfg = config = Config('others/CLIP-ES/configs/voc交互cam/base.py')

cfg.rslt_dir = ...

# * 设置模板。
cfg.templates = ['a photo of the small {}.']
