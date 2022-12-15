from prefect import flow

# custom imports
from src.pipeline.inference import megadetector_detect


@flow(
    name="MegaDetector Pipeline", 
    description="This flow implements an inference pipeline using the MegaDetector",
    log_prints=True
)
def megadetector_pipeline(
    WEIGHTS_PATH: str = "model_weights/md_v5a.0.0.pt",
    INFERENCE_SIZE: int = 1280,
    DISABLE_PBAR: bool = False,
):

    # Extract ----------------------------------------
    
    # load checkpoint

    # download images from minio or from local filesystem
    images = 0

    # Transform --------------------------------------
    results = megadetector_detect(
        weights_path=WEIGHTS_PATH,
        images=images,
        disable_pbar=DISABLE_PBAR,
        inference_size=INFERENCE_SIZE,
    )

    # Load -------------------------------------------


if __name__ == '__main__':
    megadetector_pipeline()