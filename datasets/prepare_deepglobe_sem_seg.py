#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import os
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image
from natsort import natsorted

def convert(input, output):
    img = np.asarray(Image.open(input))
    assert img.dtype == np.uint8
    img = img - 1  # 0 (ignore) becomes 255. others are shifted by 1
    Image.fromarray(img).save(output)


if __name__ == "__main__":
    _root = '/data/satellite_image_427/work/datasets/deepglobe'
    masks_train = list(natsorted((Path(_root) / 'train/masks').glob('*.png'),key = str))
    masks_valid = list(natsorted((Path(_root) / 'valid/masks').glob('*.png'),key = str))

    output_dir_train = Path(_root) / 'train/annotations_detectron2'
    output_dir_train.mkdir(parents=True, exist_ok=True)

    output_dir_valid = Path(_root) / 'valid/annotations_detectron2'
    output_dir_valid.mkdir(parents=True, exist_ok=True)

    for file in masks_train:
        output_file = output_dir_train / file.name
        convert(file, output_file)

    for file in masks_valid:
        output_file = output_dir_valid / file.name
        convert(file, output_file)        


