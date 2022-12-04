import os
import csv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

dev = 1

if dev == 0:
    DATASET_PATH = "./dataset"
else:
    DATASET_PATH = "./ds"


def get_data_from_dir(path: str) -> dict[str, np.ndarray]:
    data = {}
    for root, _, files in os.walk(path):
        for file in (pb := tqdm(files)):
            pb.set_description(f"Reading Files")
            if "DS_Store" in file:
                continue
            key = file[18:-4].lower().replace(" ", "_").replace("-", "_")
            temp = np.load(root + "/" + file, allow_pickle=True)
            data[key] = temp
        break
    return data


if __name__ == "__main__":

    data = get_data_from_dir(DATASET_PATH)
    print(data)
