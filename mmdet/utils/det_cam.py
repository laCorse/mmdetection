# Copyright (c) OpenMMLab. All rights reserved.
import bisect

import cv2
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torchvision
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from pytorch_grad_cam import AblationCAM, AblationLayer, EigenCAM
from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image

from mmdet.apis import init_detector
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose


class CAMWrapperModel(nn.Module):

    def __init__(self, cfg, checkpoint, score_thr, device='cuda:0'):
        super().__init__()
        self.device = device
        self.score_thr = score_thr
        self.detector = init_detector(cfg, checkpoint, device=device)
        self.cfg = self.detector.cfg
        self.input_data = None
        self.img = None

    def set_input_data(self, img):
        self.img = img
        cfg = self.cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
        data = dict(img=self.img)
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
        test_pipeline = Compose(cfg.data.test.pipeline)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        # just get the actual data from DataContainer
        data['img_metas'] = [
            img_metas.data[0] for img_metas in data['img_metas']
        ]
        data['img'] = [img.data[0] for img in data['img']]

        if next(self.detector.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [self.device])[0]
        else:
            for m in self.detector.modules():
                assert not isinstance(
                    m, RoIPool
                ), 'CPU inference with RoIPool is not supported currently.'

        self.input_data = data

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        assert self.input_data is not None
        results = self.detector(
            return_loss=False, rescale=True, **self.input_data)[0]

        if isinstance(results, tuple):
            bbox_result, segm_result = results
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = results, None

        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)

        if self.score_thr > 0:
            assert bboxes is not None and bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > self.score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            if segms is not None:
                segms = segms[inds, ...]
        return [{"bboxes": bboxes, 'labels': labels, 'segms': segms}]


def backbone_or_neck_reshape_transform(feats, max_shape=(20, 20)):
    if isinstance(feats, torch.Tensor):
        feats = [feats]

    max_h = max([im.shape[-2] for im in feats])
    max_w = max([im.shape[-1] for im in feats])
    max_size = (min(max_h, max_shape[0]), min(max_w, max_shape[1]))

    activations = []
    for feat in feats:
        activations.append(
            torch.nn.functional.interpolate(
                torch.abs(feat), max_size, mode='bilinear'))

    activations = torch.cat(activations, axis=1)
    return activations


class AblationLayerBackboneOrNeck(AblationLayer):

    def __init__(self):
        super(AblationLayerBackboneOrNeck, self).__init__()
        self.activations = None

    def set_next_batch(self, input_batch_index, activations,
                       num_channels_to_ablate):
        """Extract the next batch member from activations, and repeat it
        num_channels_to_ablate times."""
        if isinstance(activations, torch.Tensor):
            return super(AblationLayerBackboneOrNeck,
                         self).set_next_batch(input_batch_index, activations,
                                              num_channels_to_ablate)

        self.activations = []
        for activation in activations:
            activation = activation[
                input_batch_index, :, :, :].clone().unsqueeze(0)
            self.activations.append(
                activation.repeat(num_channels_to_ablate, 1, 1, 1))

    def __call__(self, x):
        """Go over the activation indices to be ablated, stored in
        self.indices.

        Map between every activation index to the tensor in the Ordered Dict
        from the FPN layer.
        """
        result = self.activations

        if isinstance(result, torch.Tensor):
            return super(AblationLayerBackboneOrNeck, self).__call__(x)

        channel_cumsum = np.cumsum([r.shape[1] for r in result])
        num_channels_to_ablate = result[0].size(0)  # batch
        for i in range(num_channels_to_ablate):
            # 第几层的第几个通道
            pyramid_layer = bisect.bisect_right(channel_cumsum,
                                                self.indices[i])
            if pyramid_layer > 0:
                index_in_pyramid_layer = self.indices[i] - channel_cumsum[
                    pyramid_layer - 1]
            else:
                index_in_pyramid_layer = self.indices[i]
            result[pyramid_layer][i, index_in_pyramid_layer, :, :] = -1000
        return result


class DetCam:

    def __init__(self,
                 cam_method,
                 model,
                 target_layers,
                 ablation_layer=None,
                 reshape_transform=None,
                 batch_size=1,
                 ratio_channels_to_ablate=0.1):
        if cam_method == 'ablationcam':
            self.cam = AblationCAM(
                model,
                target_layers,
                use_cuda=True if 'cuda' in model.device else False,
                reshape_transform=reshape_transform,
                batch_size=batch_size,
                ablation_layer=ablation_layer,
                ratio_channels_to_ablate=ratio_channels_to_ablate)
        elif cam_method == 'eigencam':
            self.cam = EigenCAM(
                model,
                target_layers,
                use_cuda=True if 'cuda' in model.device else False,
                reshape_transform=reshape_transform,
            )
        else:
            raise NotImplementedError()

        self.classes = model.detector.CLASSES
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def __call__(self, img, targets, aug_smooth=False, eigen_smooth=False):
        img = torch.from_numpy(img)[None].permute(0, 3, 1, 2)
        return self.cam(img, targets, aug_smooth, eigen_smooth)[0, :]

    def renormalize_cam_in_bounding_boxes(self, image, boxes, labels,
                                          grayscale_cam):
        """Normalize the CAM to be in the range [0, 1] inside every bounding
        boxes, and zero outside of the bounding boxes."""
        boxes = boxes.astype(np.int32)

        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)

        images = []
        for x1, y1, x2, y2 in boxes:
            img = renormalized_cam * 0
            img[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2,
                                                              x1:x2].copy())
            images.append(img)

        renormalized_cam = np.max(np.float32(images), axis=0)
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(
            image / 255, renormalized_cam, use_rgb=False)
        image_with_bounding_boxes = self._draw_boxes(
            boxes, labels, eigencam_image_renormalized)
        return image_with_bounding_boxes

    def _draw_boxes(self, boxes, labels, image):
        for i, box in enumerate(boxes):
            label = labels[i]
            color = self.COLORS[label]
            cv2.rectangle(image, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), color, 2)
            cv2.putText(
                image,
                self.classes[label], (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                lineType=cv2.LINE_AA)
        return image


class DetBoxScoreTarget:

    def __init__(self,
                 bboxes,
                 labels,
                 segms=None,
                 match_iou_thr=0.5,
                 device='cuda:0'):
        assert len(bboxes) == len(labels)
        self.focal_bboxes = torch.from_numpy(bboxes).to(device=device)
        self.focal_labels = labels
        if segms is not None:
            assert len(bboxes) == len(segms)
            self.focal_segms = torch.from_numpy(segms).to(device=device)
        else:
            self.focal_segms = [None] * len(labels)
        self.match_iou_thr = match_iou_thr

        self.device = device

    def __call__(self, results):
        output = torch.tensor([0], device=self.device)
        if len(results["bboxes"]) == 0:
            return output

        pred_bboxes = torch.from_numpy(results["bboxes"]).to(self.device)
        pred_labels = results["labels"]
        pred_segms = torch.from_numpy(results["segms"]).to(self.device)

        for focal_box, focal_label, focal_segm in zip(self.focal_bboxes,
                                                      self.focal_labels,
                                                      self.focal_segms):
            ious = torchvision.ops.box_iou(focal_box[None],
                                           pred_bboxes[..., :4])
            index = ious.argmax()
            if ious[0, index] > self.match_iou_thr and pred_labels[
                    index] == focal_label:
                # TODO: Adaptive adjustment of weights based on algorithms
                score = ious[0, index] + pred_bboxes[..., 4][index]

                if focal_segm is not None and pred_segms[index] is not None:
                    segms_score = (focal_segm * pred_segms[index]).sum() / (
                        focal_segm.sum() + pred_segms[index].sum() + 1e-7)
                output = output + score + segms_score
        return output
