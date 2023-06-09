import sys
import os
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.append(f'{os.getcwd()}/ultralytics')
sys.path.append(f'{os.getcwd()}/ZoeDepth')

import cv2
import torch
from PIL import Image

from ultralytics import YOLO  # noqa
from ultralytics.yolo.utils.plotting import colors  # noqa
from zoedepth.utils.misc import colorize  # noqa

output_prefix = Path('outputs')


class Detection:
    def __init__(self, model):
        self.model = self.get_detection_model(model)
        self.result = None

    def __call__(self, src, save=False):
        self.result = self.model(src)[0]
        if save:
            self.save()
        return self.result

    def plot(self, label=True):
        assert self.result is not None, 'prediction does not exist'
        return self.result.plot(labels=label)

    def save(self):
        assert self.result is not None, 'prediction does not exist'
        cv2.imwrite(str(output_prefix / 'detection_output.png'), self.plot())
        print('detection result saved as `detection_output.png`')

    @staticmethod
    def get_detection_model(model):
        if not model.endswith('pt'):
            model += '.pt'
        return YOLO(model)


class Depth:
    def __init__(self, model):
        self.model = self.get_depth_model(model)
        self.result = None

    def __call__(self, src, save=False):
        image = Image.open(src)
        self.result = self.model.infer_pil(image)
        if save:
            self.save()
        return self.result

    def plot(self):
        assert self.result is not None, 'prediction does not exist'
        return colorize(self.result)

    def save(self):
        assert self.result is not None, 'prediction does not exist'
        cv2.imwrite(str(output_prefix / 'depth_output.png'), self.plot())
        print('depth result saved as `depth_output.png`')

    @staticmethod
    def get_depth_model(model):
        repo = "isl-org/ZoeDepth"
        model_zoe_nk = torch.hub.load(repo, model, pretrained=True, trust_repo=True)
        return model_zoe_nk


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--source', default='assets/Puppies.jpg', help='source')
    parser.add_argument('--od', default='model/yolov8x-voc-best.pt', help='object detection model')
    parser.add_argument('--de', default='ZoeD_N', help='depth estimation model')
    parser.add_argument('--save', action='store_true', help='save results')

    args = parser.parse_args()

    global output_prefix
    output_prefix = output_prefix / Path(args.source).stem
    output_prefix.mkdir(parents=True, exist_ok=True)

    return args


def save_image(image, f_name):
    name = Path(f_name).stem
    cv2.imwrite(str(output_prefix / f_name), image)
    print(f'{name} result saved as `{f_name}`')


def save_images(name, images):
    classes = defaultdict(int)
    for img in images:
        c = img['cls']
        classes[c] += 1
        cv2.imwrite(str(output_prefix / f'{name[c]}_{str(classes[c]).zfill(4)}_origin.png'), img['original_img'])
        cv2.imwrite(str(output_prefix / f'{name[c]}_{str(classes[c]).zfill(4)}_depth.png'), img['depth_img'])
        cv2.imwrite(str(output_prefix / f'{name[c]}_{str(classes[c]).zfill(4)}_mask.png'), img['mask'])
        cv2.imwrite(str(output_prefix / f'{name[c]}_{str(classes[c]).zfill(4)}_gray.png'), img['gray'])
        cv2.imwrite(str(output_prefix / f'{name[c]}_{str(classes[c]).zfill(4)}_blur.png'), img['blur'])
        cv2.imwrite(str(output_prefix / f'{name[c]}_{str(classes[c]).zfill(4)}_thresh_image.png'), img['thresh_image'])
        cv2.imwrite(str(output_prefix / f'{name[c]}_{str(classes[c]).zfill(4)}_edge.png'), img['edge'])
        cv2.imwrite(str(output_prefix / f'{name[c]}_{str(classes[c]).zfill(4)}_closed_edge.png'), img['closed_edge'])
        cv2.imwrite(str(output_prefix / f'{name[c]}_{str(classes[c]).zfill(4)}_contours_image.png'), img['contours_image'])


__all__ = 'Detection', 'Depth', 'parse_args', 'save_images', 'save_image', 'colors'
