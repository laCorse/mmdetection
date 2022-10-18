_base_ = './ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py'

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

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/ssd/ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth'  # noqa
