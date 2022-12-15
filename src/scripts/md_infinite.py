import os
import time
import sys

from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torchvision

import cv2



def load_model(path):
    # load model
    model = torch.hub.load('ultralytics/yolov5:v6.2', 'custom', path)
    model.eval()
    
    return model

def load_data(image_dir_path):
    # load images
    image_dir = Path(image_dir_path)
    sample_images = [str(f) for f in image_dir.glob('*.jpg')]
    
    return sample_images

def inference(images: list, model: object):
    for f in tqdm(images, disable=False):
        im2 = cv2.imread(f)[:,:,::-1]  # OpenCV image (BGR to RGB)
        results = model([im2], size=1280) # batch of images
    
    return results

def main():
    path = './model_weights/md_v5a.0.0.pt'
    image_dir_path = './images/camtraps/'
    
    while True:
        try:
            images = load_data(image_dir_path)
            model = load_model(path)
            _ = inference(images, model)
            del model
            del images
            torch.cuda.empty_cache()
        except Exception as e:
            print(e, 'Could not perform inference')
        print('\nWaiting ...')
        time.sleep(10)

        
if __name__ == '__main__':
    main()