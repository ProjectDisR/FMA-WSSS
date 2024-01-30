#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Xiaobo Yang
@Contact : hal_42@zju.edu.cn
@Time    : 2023/8/29 22:21
@File    : generalized_cross_entropy.py
@Software: PyCharm
@Desc    : 
"""
import torch
from torch import nn
import torch.nn.functional as F

from mmseg.registry import MODELS


@MODELS.register_module()
class GCELoss(nn.Module):
    def __init__(self,
                 q=0.7,
                 class_weight=None,
                 loss_weight=1.0,
                 ignore_index=255,
                 reduction='mean',
                 loss_name: str = 'loss_gce',
                 **_
                 ):
        super(GCELoss, self).__init__()

        self.q = q

        if class_weight is not None:
            raise NotImplementedError("class_weight is not implemented in GCELoss")

        if reduction != 'mean':
            raise NotImplementedError("reduction is not implemented in GCELoss")

        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

        self._loss_name = loss_name

    def forward(self, pred, labels, *_, **__):
        num_classes = pred.shape[1]
        pred = F.softmax(pred, dim=1)

        ignore_mask = labels == self.ignore_index
        labels = labels.clone()
        labels[ignore_mask] = 0

        label_one_hot = F.one_hot(labels, num_classes).to(pred).permute(0, 3, 1, 2)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q

        return self.loss_weight * loss[~ignore_mask].mean()

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
