import os
import csv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

dev = 0

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

    variances = {}

    for root, _, files in os.walk("."):
        if "class_variances" not in "\t".join(files):
            data = get_data_from_dir(DATASET_PATH)
            for key in (pb := tqdm(data.keys())):
                pb.set_description("Calculating Variance")
                var = np.var(data[key])
                variances[key] = var
            with open("class_variances.csv", "w+") as cv:
                writer = csv.writer(cv)
                a = [[key, val] for key, val in variances.items()]
                writer.writerows(a)
            break
        else:
            print("Reading from file class_variances.csv")
            with open("class_variances.csv") as cv:
                for line in cv.readlines():
                    key = line.split(",")[0]
                    val = float(line.split(",")[1])
                    variances[key] = val
            break

    plt.plot(variances.values(), "o-", label="HAHAHAHAHAHAH")
    plt.show()
