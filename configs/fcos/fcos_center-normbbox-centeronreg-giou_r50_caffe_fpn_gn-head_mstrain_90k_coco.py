_base_ = 'fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py'

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[60000, 80000])

data = dict(workers_per_gpu=4)

# Runner type
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=90000)

checkpoint_config = dict(interval=10000)
evaluation = dict(interval=10000)

log_config = dict(interval=20)
