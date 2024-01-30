#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/2/21 20:41
@File    : par_run.py
@Software: PyCharm
@Desc    : 
"""
import os
import os.path as osp
import argparse
import subprocess
import sys
from typing import Dict, Any
import multiprocessing as mp

from alchemy_cat.py_tools.param_tuner import Cfg2TuneRunner
from alchemy_cat.torch_tools import allocate_cuda_by_group_rank

sys.path = ['others/CLIP-ES', *sys.path]  # noqa: E402


class InteractiveCamsNoGather(Cfg2TuneRunner):

    @staticmethod
    def work(pkl_idx_cfg_pkl_cfg_rslt_dir):

        pkl_idx, (cfg_pkl, cfg_rslt_dir) = pkl_idx_cfg_pkl_cfg_rslt_dir

        # * 根据分到的配置，训练网络。
        if osp.isdir(cfg_rslt_dir):
            print(f"{cfg_rslt_dir}存在，跳过{cfg_pkl}。")
        else:
            # * 找到当前应当使用的CUDA设备，并等待当前CUDA设备空闲。
            _, env_with_current_cuda = allocate_cuda_by_group_rank(pkl_idx, 1, block=True, verbosity=True)

            # * 在当前设备上执行训练。
            subprocess.run([sys.executable, 'others/CLIP-ES/tasks/interactive_cams_voc12/run.py',
                            '-s', '0', '-c', cfg_pkl],
                           check=False, env=env_with_current_cuda)

    def gather_metric(self, cfg_rslt_dir, run_rslt, param_comb) -> Dict[str, Any]:
        raise NotImplementedError


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str)
args = parser.parse_args()

runner = InteractiveCamsNoGather(args.config,
                                 config_root='others/CLIP-ES/configs',
                                 experiment_root='experiment/others/CLIP-ES',
                                 pool_size=len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
runner.tuning()
