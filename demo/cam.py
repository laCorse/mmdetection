import cv2
from pytorch_grad_cam import AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image

from mmdet.apis import init_detector
import torchvision

import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose


class RetinaNetBoxScoreTarget:
    """ For every original detected bounding box specified in "bounding boxes",
        assign a score on how the current bounding boxes match it,
            1. In IOU
            2. In the classification score.
        If there is not a large enough overlap, or the category changed,
        assign a score of 0.

        The total score is the sum of all the box scores.
    """

    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()

        if len(model_outputs["bboxes"]) == 0:
            return output

        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()
                out_box = torch.from_numpy(model_outputs["bboxes"][..., :4]).cuda()

            ious = torchvision.ops.box_iou(box, out_box)
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and model_outputs["labels"][index] == label:
                score = ious[0, index] + torch.tensor(model_outputs["bboxes"][..., 4][index])
                output = output + score
        return output


import torch.nn as nn


class RetinaNetHackModel(nn.Module):
    def __init__(self, model, imgs):
        super().__init__()
        self.model = model

        if isinstance(imgs, (list, tuple)):
            self.is_batch = True
        else:
            imgs = [imgs]
            self.is_batch = False

        cfg = model.cfg
        device = next(model.parameters()).device  # model device

        if isinstance(imgs[0], np.ndarray):
            cfg = cfg.copy()
            # set loading pipeline type
            cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
        test_pipeline = Compose(cfg.data.test.pipeline)

        datas = []
        for img in imgs:
            # prepare data
            if isinstance(img, np.ndarray):
                # directly add img
                data = dict(img=img)
            else:
                # add information into dict
                data = dict(img_info=dict(filename=img), img_prefix=None)
            # build the data pipeline
            data = test_pipeline(data)
            datas.append(data)

        data = collate(datas, samples_per_gpu=len(imgs))
        # just get the actual data from DataContainer
        data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
        data['img'] = [img.data[0] for img in data['img']]

        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
        else:
            for m in model.modules():
                assert not isinstance(
                    m, RoIPool
                ), 'CPU inference with RoIPool is not supported currently.'

        self.data = data

    def __call__(self, *args, **kwargs):
        with torch.no_grad():
            results = self.model(return_loss=False, rescale=True, **self.data)
            if not self.is_batch:
                results = results[0]
            bboxes = np.vstack(results)
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(results)
            ]
            labels = np.concatenate(labels)
            return [{"bboxes": bboxes, 'labels': labels}]

    # def eval(self):
    #     pass
    #
    # def cuda(self):
    #     return self


def draw_boxes(boxes, labels, classes, image):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
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


def renormalize_cam_in_bounding_boxes(boxes, labels, image_float_np, grayscale_cam, classes):
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
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=False)
    image_with_bounding_boxes = draw_boxes([boxes], [labels], [classes], eigencam_image_renormalized)
    return image_with_bounding_boxes


if __name__ == '__main__':
    config = '../configs/retinanet/retinanet_r50_fpn_1x_coco.py'
    checkpoint = 'retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'

    device = 'cuda:0'
    img = 'dog.jpg'  # 图片

    model = init_detector(config, checkpoint, device=device)

    img_data1 = cv2.imread(img)
    img_data = torch.from_numpy(img_data1)[None].permute(0, 3, 1, 2)

    model = RetinaNetHackModel(model, img)

    result = model(img_data)[0]  # 1=fake

    # show
    # show_result_pyplot(
    #     model,
    #     img,
    #     result,
    #     score_thr=0.3)

    # 可视化分值最高的
    bboxes = result['bboxes']
    labels = result['labels']
    index = bboxes[..., -1].argmax(-1)
    bbox = bboxes[index, :4]  # xyxy
    label = labels[index]  # xyxy

    # 只可视化第一个框
    targets = [RetinaNetBoxScoreTarget(labels=[label], bounding_boxes=[bbox])]

    target_layers = [model.model.backbone.layer3]

    cam = AblationCAM(model,
                      target_layers,
                      use_cuda=True,
                      reshape_transform=None,
                      batch_size=1,
                      ratio_channels_to_ablate=0.1)

    grayscale_cam = cam(img_data, targets=targets)[0, :]

    # show
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    # This will help us create a different color for each class
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    image_with_bounding_boxes = renormalize_cam_in_bounding_boxes(bbox, label, img_data1.astype(np.float32)/255, grayscale_cam,
                                                                  CLASSES[label])

    cv2.namedWindow('image', 0)
    cv2.imshow('image', image_with_bounding_boxes)
    cv2.waitKey(0)
