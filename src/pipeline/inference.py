import os
import sys
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import cv2

from prefect import task


def load_model(weights_path: str, model_config: dict) -> object:
    """
    Loads Megadetector from torch hub and assigns pretrained model weights.

    Parameters
    ----------
    weights_path : str
        path to weights .pt file

    model_config: dict
        model configurations

    Returns
    -------
    object
        pytorch model (yolov5)
    """
    model = torch.hub.load("ultralytics/yolov5:v6.2", "custom", weights_path)
    model.eval()

    # overwrite parameters
    model.conf = model_config['CONFIDENCE_THRESHOLD']
    model.iou = model_config['IOU_THRESHOLD']
    model.max_det = model_config['MAXIMUM_DETECTIONS']

    return model


@task(name="Perform inference with Megadetector", log_prints=True)
def megadetector_detect(
    weights_path: str,
    images: list,
    model_config: dict,
    inference_size: int = 1280,
    disable_pbar: bool = False,
) -> pd.DataFrame:
    """
    Inference function for MD

    Parameters
    ----------
    weights_path : str
        path where weights are stored

    images : list
        path of images or objects

    inference_size : int, optional
        inference size of images, by default 1280

    disable_pbar : bool, optional
        if true disables pbar, by default False

    Returns
    -------
    pd.DataFrame
        Outputs pandas dataframe with predicted class and bboxes
        for given set of images
    """
    # load model for inference
    model = load_model(weights_path=weights_path, model_config=model_config)

    all_results, times = [], []
    with tqdm(total=len(images)) as pbar:
        for file in tqdm(images, disable=disable_pbar):
            start_time = time.perf_counter()
            # read image
            im2 = cv2.imread(file)[:, :, ::-1]
            # forward pass
            with torch.no_grad():
                results = model([im2], size=model_config["INFERENCE_SIZE"])

            # post process output
            results = results.pandas().xyxy[0]
            results["object_name"] = file
            all_results.append(results)

            end_time = time.perf_counter() - start_time
            times.append(end_time)

            pbar.update(1)

    print(f"Inference mean Time: {np.mean(times).round(3)} seconds for {len(images)}")

    # concatenate all outputs
    all_results = pd.concat(all_results, axis=0).reset_index(drop=True)

    return all_results
