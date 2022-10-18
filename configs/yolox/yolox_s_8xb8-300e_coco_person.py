_base_ = './yolox_s_8xb8-300e_coco.py'

metainfo = {'CLASSES': ('person', ), 'PALETTE': (220, 20, 60)}

# model settings
model = dict(bbox_head=dict(num_classes=1))

train_dataloader = dict(
    dataset=dict(
        dataset=dict(
            metainfo=metainfo, ann_file='annotations/coco_face_train.json')))

val_dataloader = dict(
    dataset=dict(ann_file='annotations/coco_face_val.json', metainfo=metainfo))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=_base_.data_root +
                     'annotations/coco_face_val.json')

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'  # noqa

custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        strict_load=False,
        update_buffers=True,
        priority=49)
]
