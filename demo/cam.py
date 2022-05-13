import cv2
from pytorch_grad_cam import AblationCAM
import torch
from mmdet.utils.det_cam import CAMWrapperModel, DetBoxScoreTarget, DetCam

if __name__ == '__main__':
    config = '../configs/retinanet/retinanet_r50_fpn_1x_coco.py'
    checkpoint = '/home/hha/Downloads/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'

    device = 'cuda:0'
    img_path = 'dog.jpg'  # 图片

    cam_model = CAMWrapperModel(config, checkpoint, device=device)
    img = cv2.imread(img_path)
    cam_model.set_image(img)
    result = cam_model()[0]

    # 可视化分值最高的
    bboxes = result['bboxes']
    labels = result['labels']
    index = bboxes[..., -1].argmax(-1)
    bbox = bboxes[index, :4]  # xyxy
    label = labels[index]  # xyxy

    # 只可视化第一个框
    targets = [DetBoxScoreTarget(bboxes=[bbox], labels=[label])]
    target_layers = [cam_model.model.backbone.layer3]
    det_cam = DetCam('AblationCAM', cam_model, target_layers,
                     batch_size=1,
                     ratio_channels_to_ablate=0.1)

    grayscale_cam = det_cam(img, targets=targets)
    image_with_bounding_boxes = det_cam.renormalize_cam_in_bounding_boxes(img, bbox, label, grayscale_cam)

    cv2.namedWindow('image', 0)
    cv2.imshow('image', image_with_bounding_boxes)
    cv2.waitKey(0)
