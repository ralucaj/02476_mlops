# -*- coding: utf-8 -*-
import glob
import logging
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    data_files = glob.glob(f"{input_filepath}/train_*.npz")

    train = {"images": [], "labels": []}
    for data_file in data_files:
        with np.load(data_file) as data:
            train["images"].append(data["images"])
            train["labels"].append(data["labels"])
    test = np.load(f"{input_filepath}/test.npz")

    train["images"] = np.concatenate(train["images"], axis=0)
    train["labels"] = np.concatenate(train["labels"], axis=0)

    # Turn to dict and remove unwanted keys
    train = {
        "images": torch.tensor(train["images"]).float(),
        "labels": torch.tensor([train["labels"]]).squeeze(),
    }

    test = {
        "images": torch.tensor(test["images"]).float(),
        "labels": torch.tensor(test["labels"]).squeeze(),
    }

    torch.save(train, f"{output_filepath}/train.pt")
    torch.save(test, f"{output_filepath}/test.pt")
    return train, test


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
