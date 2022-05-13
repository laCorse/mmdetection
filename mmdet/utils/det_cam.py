import torch.nn as nn
import torch
from mmdet.apis import init_detector
import numpy as np
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmcv.ops import RoIPool
import cv2
from pytorch_grad_cam import AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
import torchvision


class CAMWrapperModel(nn.Module):
    def __init__(self, cfg, checkpoint, cfg_options=None, device='cuda:0'):
        super().__init__()
        self.device = device
        self.model = init_detector(cfg, checkpoint, device=device, cfg_options=cfg_options)
        self.cfg = self.model.cfg
        self.input_data = None
        self.img = None

    def set_image(self, img):
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
        data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
        data['img'] = [img.data[0] for img in data['img']]

        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [self.device])[0]
        else:
            for m in self.model.modules():
                assert not isinstance(
                    m, RoIPool
                ), 'CPU inference with RoIPool is not supported currently.'

        self.input_data = data

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        assert self.input_data is not None
        results = self.model(return_loss=False, rescale=True, **self.input_data)[0]
        bboxes = np.vstack(results)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(results)
        ]
        labels = np.concatenate(labels)
        return [{"bboxes": bboxes, 'labels': labels}]


class DetCam:
    def __init__(self, cam_method, model, target_layers, reshape_transform=None, batch_size=1,
                 ratio_channels_to_ablate=0.1):
        self.cam = AblationCAM(model,
                               target_layers,
                               use_cuda=True,
                               reshape_transform=reshape_transform,
                               batch_size=batch_size,
                               ratio_channels_to_ablate=ratio_channels_to_ablate)
        self.classes = model.model.CLASSES
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def __call__(self, img, targets):
        img = torch.from_numpy(img)[None].permute(0, 3, 1, 2)
        return self.cam(img, targets)[0, :]

    def renormalize_cam_in_bounding_boxes(self, image, boxes, labels, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1]
        inside every bounding boxes, and zero outside of the bounding boxes. """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        images = []

        x1, y1, x2, y2 = boxes.astype(np.int)
        img = renormalized_cam * 0
        img[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        images.append(img)

        renormalized_cam = np.max(np.float32(images), axis=0)
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image / 255, renormalized_cam, use_rgb=False)
        image_with_bounding_boxes = self._draw_boxes([boxes], [labels], [self.classes[labels]],
                                                     eigencam_image_renormalized)
        return image_with_bounding_boxes

    def _draw_boxes(self, boxes, labels, classes, image):
        for i, box in enumerate(boxes):
            color = self.COLORS[labels[i]]
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color, 2
            )
            cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                        lineType=cv2.LINE_AA)
        return image


class DetBoxScoreTarget:
    def __init__(self, bboxes, labels, match_iou_thr=0.5, device='cuda:0'):
        self.focal_bboxes = bboxes
        self.focal_labels = labels
        assert isinstance(self.focal_bboxes, list)
        assert isinstance(self.focal_labels, list)
        assert len(self.focal_bboxes) == len(self.focal_labels)
        self.focal_bboxes = [torch.tensor(focal_bbox[None, :], device=device)
                             for focal_bbox in self.focal_bboxes]
        self.match_iou_thr = match_iou_thr
        self.device = device

    def __call__(self, results):
        output = torch.tensor([0], device=self.device)
        if len(results["bboxes"]) == 0:
            return output

        pred_bboxes = torch.from_numpy(results["bboxes"]).to(self.device)
        pred_labels = results["labels"]

        for focal_box, focal_label in zip(self.focal_bboxes, self.focal_labels):
            ious = torchvision.ops.box_iou(focal_box, pred_bboxes[..., :4])
            index = ious.argmax()
            if ious[0, index] > self.match_iou_thr and pred_labels[index] == focal_label:
                score = ious[0, index] + pred_bboxes[..., 4][index]
                output = output + score
        return output
