_base_ = './retinanet_r50_fpn_1x_coco.py'

metainfo = {'CLASSES': 'person', 'PALETTE': (220, 20, 60)}

model = dict(bbox_head=dict(num_classes=1))

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736), (1333, 768),
                (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        pipeline=train_pipeline,
        ann_file='annotations/coco_face_train.json'))

val_dataloader = dict(dataset=dict(ann_file='annotations/coco_face_val.json', metainfo=metainfo))
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=_base_.data_root + 'annotations/coco_face_val.json')

load_from = 'https://download.openmmlab.com/mmbenchmark/v0_models/mmdet/retinanet/retinanet_r50_fpn_lsj_200e_8x8_fp16_coco/retinanet_r50_fpn_lsj_200e_8x8_fp16_coco-87541d36.pth'  # noqa
