import os
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
import tensorflow as tf

dev = 1

if dev == 0:
    DATASET_PATH = "./dataset"
else:
    DATASET_PATH = "./ds"


def get_data_from_dir(
    path: str,
) -> tuple[tuple[list, list], dict[int, str]]:
    data = {}
    for root, _, files in os.walk(path):
        for file in (pb := tqdm(files)):
            pb.set_description(f"Reading Files")
            if "DS_Store" in file:
                continue
            key = file[18:-4].lower().replace(" ", "_").replace("-", "_")
            data[key] = np.load(root + "/" + file, allow_pickle=True)

    class_map = {idx: v for idx, v in enumerate(data.keys())}

    layer = tf.keras.layers.CategoryEncoding(
        num_tokens=len(class_map.keys()), output_mode="one_hot"
    )

    one_hot_map = {}
    for (idx, v) in enumerate(layer(class_map.keys())):
        one_hot_map[class_map[idx]] = v

    X = []
    Y = []
    for key, items in data.items():
        for item in items:
            X.append(item)
            Y.append(one_hot_map[key])

    return ((X, Y), one_hot_map)


if __name__ == "__main__":

    ((X, Y), one_hot_map) = get_data_from_dir(DATASET_PATH)
    # converting from np.array is better/performant than from list
    X, Y = tf.convert_to_tensor(np.array(X)), tf.convert_to_tensor(np.array(Y))
    print(X.shape)
    print(type(X))
    print(Y.shape)

    # print(data)
