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
from alchemy_cat.py_tools import Cfg2Tune, Param2Tune

cfg = config = Cfg2Tune(cfgs_update_at_parser=('others/CLIP-ES/configs/voc交互cam/base.py',))

cfg.rslt_dir = ...

# * 设置模板。
cfg.templates = Param2Tune(["a high contrast photo of my nice {}.",
                            "the clean example of the a hard to see {}.",
                            "a bright sculpture of many dirty {}.",
                            "the close-up embroidered of a old {}.",
                            "a cropped cartoon of the weird {}.",
                            "the centered jpeg corrupted photo of a cool {}.",
                            "a dark plushie of the new {}.",
                            "the low resolution rendering of a clean {}."])
