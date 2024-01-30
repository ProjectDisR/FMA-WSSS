#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import os.path as osp

from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import cv2
import argparse
from lxml import etree

import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode

from alchemy_cat.torch_tools import init_env
from alchemy_cat.contrib.tasks.wsss.viz import viz_cam
from alchemy_cat.acplot import col_all, BGR2RGB
from alchemy_cat.contrib.voc import VOCAug

import sys
sys.path = ['others/CLIP-ES', *sys.path]  # noqa: E402

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import scale_cam_image
import clip
from clip.model import CLIP
from utils import parse_xml_to_dict, scoremap2bbox

BICUBIC = InterpolationMode.BICUBIC


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)  # LND -> NLD
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))  # (N, H, W, D)

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)  # (N, D, H, W)
    return result


def zeroshot_classifier(cls_names, templs, m):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in cls_names:
            texts = [templ.format(classname) for templ in templs]  # format with class
            texts = clip.tokenize(texts).to(device)  # tokenize
            class_embeddings = m.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.t()


class ClipOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


def read_voc_img_info(img_path: str, ori2new_class_names: dict[str, str], new_class_names: list[str])\
        -> tuple[tuple[int, int], list[str], list[int]]:
    # * 读取图片xml配置。
    xmlfile = img_path.replace('/JPEGImages', '/Annotations')
    xmlfile = xmlfile.replace('.jpg', '.xml')
    with open(xmlfile) as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)  # etree包 读取xml文件
    data = parse_xml_to_dict(xml)["annotation"]

    # * 读取原始尺寸。
    ori_height = int(data['size']['height'])
    ori_width = int(data['size']['width'])

    # * 读取标签。
    label_list = set()
    for obj in data["object"]:
        label_list |= set(ori2new_class_names[obj['name']])
    label_list = list(label_list)
    assert len(label_list) > 0
    label_list.sort(key=lambda l: new_class_names.index(l))

    label_id_list = [new_class_names.index(label) for label in label_list]

    return (ori_height, ori_width), label_list, label_id_list


def _convert_image_to_rgb(img):
    return img.convert("RGB")


def _transform_resize(h, w):
    return Compose([
        Resize((h, w), interpolation=BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def img_ms_and_flip(img_path, ori_height, ori_width, scales=(1.0,), flip=True, patch_size=16):
    ori_img = Image.open(img_path)

    all_imgs = []
    for scale in scales:
        preprocess = _transform_resize(int(np.ceil(scale * int(ori_height) / patch_size) * patch_size),
                                       int(np.ceil(scale * int(ori_width) / patch_size) * patch_size))
        img = preprocess(ori_img)  # 缩放（调整到patch整数倍尺寸）、CHW、归一化。
        all_imgs.append(img)
        if flip:
            all_imgs.append(torch.flip(img, [-1]))

    return all_imgs


def ms_cam(ms_imgs: list[torch.Tensor], ori_height: int, ori_width: int, label_id_list: list[int],
           model: CLIP, cam: GradCAM,
           fg_features: torch.Tensor, bg_features: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    # * 选出互斥前景配置下的前景特征+背景特征参与计算。
    if not cfg.indep_fg:
        proxies = torch.cat((fg_features[label_id_list, :], bg_features), dim=0)

    # * 准备CAM容器。
    highres_cam_all_scales = []
    refined_cam_all_scales = []

    # * 逐尺度推理。
    """
                   scale0     scale1
        label0      CAM        CAM
        label1      CAM        CAM（全为原图尺寸）
    """
    for img in ms_imgs:
        # * 编码图片特征。
        img = img.unsqueeze(0)  # (1, 3, H, W)
        img_h, img_w = img.shape[-2], img.shape[-1]
        img_features, att_weights = model.encode_image(img.to(next(model.parameters()).device), img_h, img_w)

        # * 记录每个标签的CAM。
        highres_cam_to_save = []
        refined_cam_to_save = []

        # * 逐标签推理、后处理CAM。
        """
                  high_res    refined
        label0      CAM        CAM
        label1      CAM        CAM（全为原图尺寸）
        """
        for idx in range(len(label_id_list)):
            # * 选出独立前景特征+背景特征参与计算。
            if cfg.indep_fg:
                proxies = torch.cat((fg_features[label_id_list[idx:idx+1], :], bg_features), dim=0)

            grayscale_cam, logits_per_image, attn_weight_last = cam(input_tensor=[img_features, proxies,
                                                                                  img_h, img_w],
                                                                    targets=[ClipOutputTarget(0 if cfg.indep_fg
                                                                                              else idx)],
                                                                    target_size=None)  # (ori_width, ori_height))
            grayscale_cam = grayscale_cam[0, :]

            grayscale_cam_highres = cv2.resize(grayscale_cam, (ori_width, ori_height))
            highres_cam_to_save.append(torch.tensor(grayscale_cam_highres))

            if idx == 0:
                att_weights.append(attn_weight_last)
                attn_weight = [aw[:, 1:, 1:] for aw in att_weights]  # (b, hxw, hxw)
                attn_weight = torch.stack(attn_weight, dim=0)[-8:]
                attn_weight = torch.mean(attn_weight, dim=0)
                attn_weight = attn_weight[0].cpu().detach()
            attn_weight = attn_weight.float()

            box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.4, multi_contour_eval=True)
            aff_mask = torch.zeros((grayscale_cam.shape[0], grayscale_cam.shape[1]))
            for i_ in range(cnt):
                x0_, y0_, x1_, y1_ = box[i_]
                aff_mask[y0_:y1_, x0_:x1_] = 1

            aff_mask = aff_mask.view(1, grayscale_cam.shape[0] * grayscale_cam.shape[1])  # 只聚合连通域BBox内像素。
            aff_mat = attn_weight

            trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
            trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

            for _ in range(2):
                trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
                trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
            trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

            for _ in range(1):
                trans_mat = torch.matmul(trans_mat, trans_mat)

            trans_mat = trans_mat * aff_mask  # sink-horn、对称、幂次、加掩码后的聚合矩阵。

            cam_to_refine = torch.FloatTensor(grayscale_cam)
            cam_to_refine = cam_to_refine.view(-1, 1)

            # (n,n) * (n,1)->(n,1)
            cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(img_h // 16, img_w // 16)
            cam_refined = cam_refined.cpu().numpy().astype(np.float32)
            cam_refined_highres = scale_cam_image([cam_refined], (ori_width, ori_height))[0]
            refined_cam_to_save.append(torch.tensor(cam_refined_highres))  # 记录优化后CAM。

        highres_cam_all_scales.append(torch.stack(highres_cam_to_save, dim=0))  # list[tensor[2, H, W]]
        refined_cam_all_scales.append(torch.stack(refined_cam_to_save, dim=0))

    highres_cam_all_scales = torch.stack(highres_cam_all_scales, dim=0).mean(dim=0)  # tensor[2, H, W]
    refined_cam_all_scales = torch.stack(refined_cam_all_scales, dim=0).mean(dim=0)

    return highres_cam_all_scales.numpy(), refined_cam_all_scales.numpy()


# * 读取数据集、pretrain、实验结果位置。
parser = argparse.ArgumentParser(description='')
parser.add_argument('--img_root', type=str, default='datasets/VOC2012/JPEGImages')
parser.add_argument('--split_file', type=str, default='others/CLIP-ES/voc12/train_aug.txt')
parser.add_argument('--cam_out_dir', type=str, default=f'experiment/others/CLIP-ES')
parser.add_argument('--model', type=str, default='pretrains/CLIP/ViT-B-16.pt')
parser.add_argument('-c', '--cfg', type=str)
parser.add_argument('-s', '--show_viz', default=0, type=int)
args = parser.parse_args()

# * 环境初始化。
device, cfg = init_env(is_cuda=True,
                       is_benchmark=False,
                       is_train=True,
                       config_path=args.cfg,
                       config_root='others/CLIP-ES/configs',
                       experiments_root=args.cam_out_dir,
                       rand_seed=False,
                       cv2_num_threads=-1,
                       verbosity=True,
                       log_stdout=True,
                       reproducibility=False,
                       is_debug=False)

# * 对应建立我的VOC数据集，便于可视化。
dt = VOCAug('datasets', split='train_aug')

# * 加载CLIP模型。
model, _ = clip.load(args.model, device=device)

# * 建立GradCAM。
target_layers = [model.visual.transformer.resblocks[-1].ln_1]  # 记录最后一层首个LN输出上的激活值和梯度。
cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

# * 建立输出目录，并保存配置文件。
os.makedirs(cfg.rslt_dir, exist_ok=True)

highres_cams_dir = osp.join(cfg.rslt_dir, 'highres_cams')
refined_cams_dir = osp.join(cfg.rslt_dir, 'refined_cams')
os.makedirs(highres_cams_dir, exist_ok=True)
os.makedirs(refined_cams_dir, exist_ok=True)

# * 编码文本特征。
# * 获取类别名。
fg_bg_names = cfg.new_class_names + cfg.bg_names

# ** 将模板转为多模板分类别格式。
# 模板可以是''，表示单模板全类别；可以是['', ...]，表示多模板全类别；可以是[['', ...], ...]，表示多模板分类别。
match cfg.templates:
    case str(template):
        templates = [[template]] * len(fg_bg_names)
    case [str(), *_] as template:
        templates = [template] * len(fg_bg_names)
    case [[str(), *_], *_] as template:
        templates = template
    case _:
        raise RuntimeError(f"不支持的{cfg.templates=}。")
# ** 获取各类别的ensemble模板。
'''
    cls A  cls B  cls C
tl0  *
tl1         ...
tl2                 *
     tfA    tfB    tfC
'''
text_features = []
for class_name, template in zip(fg_bg_names, templates):
    text_features.append(zeroshot_classifier([class_name], template, model))
text_features = torch.cat(text_features, dim=0)
fg_features, bg_features = text_features[:len(cfg.new_class_names)], text_features[len(cfg.new_class_names):]

# * 对每张图片，计算其关于文本特征的CAM。
for img_name in tqdm(cfg.img_names, desc="处理", unit="张", dynamic_ncols=True):
    img_path = osp.join(args.img_root, img_name + '.jpg')

    # * 读取图片元信息。
    (ori_height, ori_width), label_list, label_id_list = read_voc_img_info(img_path,
                                                                           cfg.ori2new_class_names,
                                                                           cfg.new_class_names)

    # * 读取原图及标签。
    _, ori_img, ori_label = dt.get_by_img_id(img_name)
    ori_img = BGR2RGB(ori_img)

    # * 读取并预处理图片。
    ms_imgs = img_ms_and_flip(img_path, ori_height, ori_width, scales=cfg.scales, flip=cfg.flip,
                              patch_size=model.visual.patch_size)

    # * 获取当前图片的CAM。
    highres_cams, refined_cams = ms_cam(ms_imgs, ori_height, ori_width, label_id_list,
                                        model, cam,
                                        fg_features, bg_features)

    # * 作图并保存CAM。
    fig = plt.figure(dpi=600)

    # ** 可视化highres CAM。
    viz_cam(fig=fig,
            img_id=img_name, img=ori_img, label=ori_label,
            cls_in_label=np.ones(len(label_list) + 1, dtype=np.int32),
            cam=highres_cams, cls_names=['dummy'] + label_list,
            gamma=1., blend_alpha=.5, get_row_col=cfg.show_row_col or col_all)

    if args.show_viz:
        fig.show()

    # ** 保存CAM可视化结果。
    fig.savefig(osp.join(highres_cams_dir, f'{img_name}.png'), bbox_inches='tight')
    fig.clf()

    # ** 可视化refined CAM。
    viz_cam(fig=fig,
            img_id=img_name, img=ori_img, label=ori_label,
            cls_in_label=np.ones(len(label_list) + 1, dtype=np.int32),
            cam=refined_cams, cls_names=['dummy'] + label_list,
            gamma=1., blend_alpha=.5, get_row_col=cfg.show_row_col or col_all)

    if args.show_viz:
        fig.show()

    # ** 保存CAM可视化结果。
    fig.savefig(osp.join(refined_cams_dir, f'{img_name}.png'), bbox_inches='tight')
    fig.clf()
