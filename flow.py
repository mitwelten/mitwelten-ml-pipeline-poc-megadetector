
import os
import yaml
import json

import pandas as pd

from prefect import flow

# custom imports
from src.pipeline.inference import megadetector_detect
from src.pipeline.clients import get_db_client, get_minio_client


@flow(
    name="MegaDetector Pipeline",
    description="This flow implements an inference pipeline using the MegaDetector",
    log_prints=True
)
def megadetector_pipeline(
    WEIGHTS_PATH: str = "model_weights/md_v5a.0.0.pt",
    SOURCE_CONFIG_PATH: str = "source_config.yaml",
    MODEL_CONFIG_PATH: str = "model_config.json",
    BATCHSIZE: int = 16,
    INFERENCE_SIZE: int = 1280,
    DISABLE_PBAR: bool = False,
    IS_TEST: bool = True
):

    # Load configs -----------------------------------
    if not IS_TEST:
        db_client = get_db_client(config_path=SOURCE_CONFIG_PATH)
        minio_client = get_minio_client(config_path=SOURCE_CONFIG_PATH)
    else:    
        with open(SOURCE_CONFIG_PATH, 'rb') as yaml_file:
            source_config = yaml.load(yaml_file, yaml.FullLoader)

    # load model configs
    with open(MODEL_CONFIG_PATH, 'r') as json_file:
        model_config = json.load(json_file)



    # Extract ----------------------------------------
    # load checkpoint
    if IS_TEST:
        df = pd.read_csv(source_config['SAMPLES_DF'])
        data = df[df['processed'] == False].iloc[:BATCHSIZE]
        images = data['image_path'].to_list()
    else:
        raise NotImplementedError

    # Transform --------------------------------------
    results = megadetector_detect(
        weights_path=WEIGHTS_PATH,
        images=images,
        model_config=model_config,
        disable_pbar=DISABLE_PBAR,
        inference_size=INFERENCE_SIZE,
    )

    # Load -------------------------------------------
    if IS_TEST:
        # writes results to a dataframe and updates the given 
        # checkpoint dataframe
        if os.path.isfile(source_config['RESULTS_DF']):
            previous_results = pd.read_csv(source_config['RESULTS_DF'])
            results = pd.concat([previous_results, results], axis=0)
        results.to_csv(source_config['RESULTS_DF'], index=False)
        df.loc[df['filename'].isin(data['filename']), 'processed'] = 1
        df.to_csv(source_config['SAMPLES_DF'])
    else:
        raise NotImplementedError


if __name__ == '__main__':
    megadetector_pipeline()
