from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
from loguru import logger
import tensorflow as tf
from predict_transformation_between_two_images import *
from utils import *

INDEX_COLS = ["chain_id", "i"]
PREDICTION_COLS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
REFERENCE_VALUES = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
ESTIMATED_DISTANCE_TO_SPACECRAFT = (20 + 600) / 2 # metres

MODEL = tf.keras.models.load_model('my_model.keras')

def predict_chain(chain_dir: Path):
    logger.debug(f"making predictions for {chain_dir}")
    chain_id = chain_dir.name
    image_paths = list(sorted(chain_dir.glob("*.png")))
    path_per_idx = {int(image_path.stem): image_path for image_path in image_paths}
    idxs = list(sorted(path_per_idx.keys()))

    assert idxs[0] == 0, f"First index for chain {chain_id} is not 0"
    assert (
        np.diff(idxs) == 1
    ).all(), f"Skipped image indexes found in chain {chain_id}"

    # pick out the reference image
    try:
        reference_img_path = path_per_idx[0]
        _reference_img = cv2.imread(str(reference_img_path))
    except KeyError:
        raise ValueError(f"Could not find reference image for chain {chain_id}")

    # create an empty dataframe to populate with values
    chain_df = pd.DataFrame(
        index=pd.Index(idxs, name="i"), columns=PREDICTION_COLS, dtype=float
    )

    net_returning_rotation = np.array([1, 0, 0, 0])
    net_returning_translation = np.array([0, 0, 0])
    ESTIMATED_DISTANCE_TO_SPACECRAFT = (20 + 600) / 2 # metres
    last_known_range = ESTIMATED_DISTANCE_TO_SPACECRAFT

    # make a prediction for each image
    for i, image_path in path_per_idx.items():
        if i == 0:
            predicted_values = REFERENCE_VALUES
        else:
            # Think of image h as the image before image i.
            img_h = cv2.imread(path_per_idx[i-1])
            img_i = cv2.imread(path_per_idx[i])

            # Get the raw incremental returning transformation using SIFT and RANSAC.
            raw_rotation, raw_translation = predict_transformation_between_two_images(img_i, img_h)

            # Use the pre-trained neural network to refine the incremental returning transformation.
            # TODO: Read the estimated distance to the spacecraft to the file.
            neural_network_input = np.hstack([last_known_range, raw_translation, raw_rotation])
            refined_rotation, refined_translation = MODEL.predict(neural_network_input)[0]

            # Update the net returning transformation.
            net_returning_rotation, net_returning_translation = compose_transformations(refined_rotation,
                                                                                        refined_translation,
                                                                                        net_returning_rotation,
                                                                                        net_returning_translation)[0]
            
            predicted_values = np.hstack([net_returning_translation, net_returning_rotation])
        
        chain_df.loc[i] = predicted_values

    # double check we made predictions for each image
    assert (
        chain_df.notnull().all(axis="rows").all()
    ), f"Found NaN values for chain {chain_id}"
    assert (
        np.isfinite(chain_df.values).all().all()
    ), f"Found NaN or infinite values for chain {chain_id}"

    return chain_df


@click.command()
@click.argument(
    "data_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.argument(
    "output_path",
    type=click.Path(exists=False),
)
def main(data_dir, output_path):
    data_dir = Path(data_dir).resolve()
    output_path = Path(output_path).resolve()
    assert (
        output_path.parent.exists()
    ), f"Expected output directory {output_path.parent} does not exist"

    logger.info(f"using data dir: {data_dir}")
    assert data_dir.exists(), f"Data directory does not exist: {data_dir}"

    # read in the submission format
    submission_format_path = data_dir / "submission_format.csv"
    submission_format_df = pd.read_csv(submission_format_path, index_col=INDEX_COLS)

    # copy over the submission format so we can overwrite placeholders with predictions
    submission_df = submission_format_df.copy()

    image_dir = data_dir / "images"
    chain_ids = submission_format_df.index.get_level_values(0).unique()
    for chain_id in chain_ids:
        logger.info(f"Processing chain: {chain_id}")
        chain_dir = image_dir / chain_id
        assert chain_dir.exists(), f"Chain directory does not exist: {chain_dir}"
        chain_df = predict_chain(chain_dir)
        submission_df.loc[chain_id] = chain_df.values

    submission_df.to_csv(output_path, index=True)


if __name__ == "__main__":
    main()
