import argparse
from collections import OrderedDict

import mmcv
import numpy as np
import torch


def convert(src, dst):
    src_model = mmcv.load(src)

    dst_state_dict = OrderedDict()
    for k, v in src_model['model'].items():
        key_name_split = k.split('.')
        if 'backbone.fpn_lateral' in k:
            lateral_id = int(key_name_split[-2][-1])
            name = f'neck.lateral_convs.{lateral_id - 3}.conv.{key_name_split[-1]}'
        elif 'backbone.fpn_output' in k:
            lateral_id = int(key_name_split[-2][-1])
            name = f'neck.fpn_convs.{lateral_id - 3}.conv.{key_name_split[-1]}'
        elif 'backbone.top_block' in k:
            top_block_id = int(key_name_split[-2][-1])
            name = f'neck.fpn_convs.{top_block_id - 3}.conv.{key_name_split[-1]}'
        elif 'backbone.bottom_up.stem.conv1.norm.' in k:
            name = f'backbone.bn1.{key_name_split[-1]}'
        elif 'backbone.bottom_up.stem.conv1.' in k:
            name = f'backbone.conv1.{key_name_split[-1]}'
        elif 'backbone.bottom_up.res' in k:
            weight_type = key_name_split[-1]
            res_id = int(key_name_split[2][-1]) - 1
            # deal with short cut
            if 'shortcut' in key_name_split[4]:
                if 'shortcut' == key_name_split[-2]:
                    name = f'backbone.layer{res_id}.{key_name_split[3]}.downsample.0.{key_name_split[-1]}'
                elif 'shortcut' == key_name_split[-3]:
                    name = f'backbone.layer{res_id}.{key_name_split[3]}.downsample.1.{key_name_split[-1]}'
                else:
                    print(f'Unvalid key {k}')
            # deal with conv
            elif 'conv' in key_name_split[-2]:
                conv_id = int(key_name_split[-2][-1])
                name = f'backbone.layer{res_id}.{key_name_split[3]}.conv{conv_id}.{key_name_split[-1]}'
            # deal with BN
            elif key_name_split[-2] == 'norm':
                conv_id = int(key_name_split[-3][-1])
                name = f'backbone.layer{res_id}.{key_name_split[3]}.bn{conv_id}.{key_name_split[-1]}'
            else:
                print(f'{k} is invalid')
        elif 'head' in k:
            if key_name_split[1] == 'cls_subnet':
                fc_id = int(key_name_split[2])//2
                name = f'bbox_head.cls_convs.{fc_id}.conv.{key_name_split[-1]}'
            elif key_name_split[1] == 'bbox_subnet':
                fc_id = int(key_name_split[2])//2
                name = f'bbox_head.reg_convs.{fc_id}.conv.{key_name_split[-1]}'
            elif key_name_split[1] == 'cls_score':
                name = f'bbox_head.retina_cls.{key_name_split[-1]}'
            elif key_name_split[1] == 'bbox_pred':
                name = f'bbox_head.retina_reg.{key_name_split[-1]}'
            else:
                print(f'{k} is invalid')
        elif 'pixel_' in k or 'anchor_generator' in k:
            continue
        else:
            print(f'{k} is not converted!!')

        if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
            raise ValueError(
                'Unsupported type found in checkpoint! {}: {}'.format(
                    k, type(v)))
        if not isinstance(v, torch.Tensor):
            dst_state_dict[name] = torch.from_numpy(v)

    mmdet_model = dict(state_dict=dst_state_dict, meta=dict())
    torch.save(mmdet_model, dst)


# d2 retinanet 配置 configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml
# 权重 model_final_bfca0b.pkl, mAP 37.5
# 转换后 AnchorGenerator 的 需要 scale_major=True，否则结果对不上
def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('src', help='src detectron model path')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    # src_model = mmcv.load(args.src)
    # for k, v in src_model['model'].items():
    #     print(k)

    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
