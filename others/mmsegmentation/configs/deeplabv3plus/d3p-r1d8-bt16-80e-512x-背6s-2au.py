_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/pascal_voc12_aug_seed.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'  # ** 设置迭代次数为80k。
]

# * 设置R101 Backbone，512x512输入。
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=21),
    auxiliary_head=dict(
        num_classes=21,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2))
)

# * 设置训练数据流。
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Pad', size=crop_size),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=8,  # ** 设置Batch Size=8x2。
    num_workers=8,  # ** 增大num_workers。
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='PascalVOCDataset',
        data_root='',
        data_prefix=dict(
            img_path='data/VOC2012/JPEGImages',
            # ** 设置种子点路径。
            seg_map_path='exp_root/anns_seed/ta/ann=l2_nmsf_s1_rsw3,cam=无背头6,affed/seed'),
        ann_file='data/VOC2012/ImageSets/Segmentation/train_aug.txt',
        pipeline=train_pipeline))

# * 自动缩放学习率。
auto_scale_lr = dict(enable=True, base_batch_size=16)

# * 使用AMP。
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=optimizer,
    loss_scale='dynamic')
