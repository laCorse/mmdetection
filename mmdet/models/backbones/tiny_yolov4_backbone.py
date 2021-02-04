# -*- coding:utf-8 -*-
from collections import OrderedDict
import torch.nn as nn
from ..utils import yolo_brick as vn_layer
from ..builder import BACKBONES


@BACKBONES.register_module()
class TinyYolov4Backbone(nn.Module):
    # 用于递归导入权重
    custom_layers = (vn_layer.ResConv2dBatchLeaky,)

    def __init__(self, pretrained=False):
        super(TinyYolov4Backbone, self).__init__()

        # Network
        backbone = OrderedDict([
            ('0_convbatch', vn_layer.Conv2dBatchLeaky(3, 32, 3, 2)),
            ('1_convbatch', vn_layer.Conv2dBatchLeaky(32, 64, 3, 2)),
            ('2_convbatch', vn_layer.Conv2dBatchLeaky(64, 64, 3, 1)),
            ('3_resconvbatch', vn_layer.ResConv2dBatchLeaky(64, 32, 3, 1)),
            ('4_max', nn.MaxPool2d(2, 2)),
            ('5_convbatch', vn_layer.Conv2dBatchLeaky(128, 128, 3, 1)),
            ('6_resconvbatch', vn_layer.ResConv2dBatchLeaky(128, 64, 3, 1)),
            ('7_max', nn.MaxPool2d(2, 2)),
            ('8_convbatch', vn_layer.Conv2dBatchLeaky(256, 256, 3, 1)),
            ('9_resconvbatch', vn_layer.ResConv2dBatchLeaky(256, 128, 3, 1, return_extra=True)),
        ])

        self.layers = nn.Sequential(backbone)
        self.init_weights(pretrained)

    def __modules_recurse(self, mod=None):
        """ This function will recursively loop over all module children.
        Args:
            mod (torch.nn.Module, optional): Module to loop over; Default **self**
        """
        if mod is None:
            mod = self
        for module in mod.children():
            if isinstance(module, (nn.ModuleList, nn.Sequential)):
                yield from self.__modules_recurse(module)
            else:
                yield module

    def init_weights(self, pretrained=None):
        pass

    def forward(self, x):
        stem, extra_x = self.layers(x)
        return [stem, extra_x]
