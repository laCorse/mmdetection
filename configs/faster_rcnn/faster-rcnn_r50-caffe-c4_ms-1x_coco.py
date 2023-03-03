_base_ = './faster-rcnn_r50-caffe_c4-1x_coco.py'

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736), (1333, 768),
               (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
_base_.train_dataloader.dataset.pipeline = train_pipeline
