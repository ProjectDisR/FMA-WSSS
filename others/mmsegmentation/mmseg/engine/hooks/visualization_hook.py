# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
import mmengine.fileio as fileio
import numpy as np
import torch
from alchemy_cat.data.plugins import arr2PIL
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer


@HOOKS.register_module()
class SegVisualizationHook(Hook):
    """Segmentation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None,
                 save_prob: bool=False):
        self._visualizer: SegLocalVisualizer = \
            SegLocalVisualizer.get_current_instance()
        self.interval = interval
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')

        self.save_prob = save_prob

    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """
        if self.draw is False or mode == 'train':
            return

        if self.save_prob and mode == 'test':
            os.makedirs(prob_dir := osp.join(runner.work_dir, 'seg_preds'), exist_ok=True)
            os.makedirs(mask_dir := osp.join(runner.work_dir, 'masks'), exist_ok=True)

            for output in outputs:
                img_id = osp.splitext(osp.basename(output.img_path))[0]

                all_cls = torch.unique(output.pred_sem_seg.data).cpu().numpy()
                if all_cls[0] == 0:  # 如[0, 1, 15]。
                    fg_cls = all_cls[1:] - 1
                else:  # 如[1, 15]。
                    fg_cls = all_cls - 1

                logit = output.seg_logits.data.cpu().numpy()[all_cls, :, :]

                if hasattr(output, 'seg_probs'):
                    prob = output.seg_probs.data.cpu().numpy()[all_cls, :, :]
                else:
                    prob = output.seg_logits.data.softmax(dim=0).cpu().numpy()[all_cls, :, :]

                mask = output.pred_sem_seg.data.to(dtype=torch.uint8).squeeze().cpu().numpy()

                np.savez(osp.join(prob_dir, f'{img_id}.npz'), prob=prob, logit=logit, fg_cls=fg_cls)

                arr2PIL(mask).save(osp.join(mask_dir, f'{img_id}.png'))

        if self.every_n_inner_iters(batch_idx, self.interval):
            for output in outputs:
                img_path = output.img_path
                img_bytes = fileio.get(
                    img_path, backend_args=self.backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                window_name = f'{mode}_{osp.basename(img_path)}'

                self._visualizer.add_datasample(
                    window_name,
                    img,
                    data_sample=output,
                    show=self.show,
                    wait_time=self.wait_time,
                    step=runner.iter)
