_base_ = './retinanet_r50_fpn_lsj_100e_coco.py'

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        frozen_stages=-1,
        norm_eval=False,
        norm_cfg=norm_cfg,
        init_cfg=None,
        style='caffe'))
