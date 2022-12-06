import os
import cProfile
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
) -> tuple[tuple[list, list], dict[str, int]]:
    data = {}
    for root, _, files in os.walk(path):
        for file in (pb := tqdm(files)):
            pb.set_description(f"Reading Files")
            if "DS_Store" in file:
                continue
            key = file[18:-4].lower().replace(" ", "_").replace("-", "_")
            data[key] = np.load(root + "/" + file, allow_pickle=True)
            data[key] = tf.reshape(data[key], shape=(data[key].shape[0], 28, 28, 1))

    class_map = {v: idx for idx, v in enumerate(data.keys())}

    # layer = tf.keras.layers.CategoryEncoding(
    #     num_tokens=len(class_map.keys()), output_mode="one_hot"
    # )
    #
    # one_hot_map = {}
    # for (idx, v) in enumerate(layer(class_map.keys())):
    #     one_hot_map[class_map[idx]] = v

    X = []
    Y = []
    for key, items in data.items():
        for item in items:
            X.append(item)
            Y.append([class_map[key]])

    return ((X, Y), class_map)


if __name__ == "__main__":

    pr = cProfile.Profile()
    pr.enable()
    # getting data
    ((X, Y), one_hot_map) = get_data_from_dir(DATASET_PATH)
    # converting from np.array is better/performant than from list
    X, Y = tf.convert_to_tensor(np.array(X)), tf.convert_to_tensor(np.array(Y))

    # shuffling
    indices = tf.range(start=0, limit=tf.shape(X)[0])
    shuffled_indices = tf.random.shuffle(indices)
    X, Y = tf.gather(X, shuffled_indices), tf.gather(Y, shuffled_indices)
    input_shape = (tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2], 1)

    train_X, train_Y, val_X, val_Y, test_X, test_Y = 0, 0, 0, 0, 0, 0

    # splitting data
    if dev == 1:
        train_X, train_Y = (X[:640], Y[:640])

        val_X, val_Y = (X[640:160], Y[640:160])

        test_X, test_Y = (X[800:], Y[800:])
    else:
        train_X, train_Y = (
            X[: int(int(int(tf.shape(X)[0]) * 0.8) * 0.8)],
            Y[: int(int(int(tf.shape(Y)[0]) * 0.8) * 0.8)],
        )

        val_X, val_Y = (
            X[
                int(int(int(tf.shape(X)[0]) * 0.8) * 0.8) : int(
                    int(tf.shape(X)[0]) * 0.8
                )
            ],
            Y[
                int(int(int(tf.shape(Y)[0]) * 0.8) * 0.8) : int(
                    int(tf.shape(X)[0]) * 0.8
                )
            ],
        )

        test_X, test_Y = (
            X[int(int(tf.shape(X)[0]) * 0.8) :],
            Y[int(int(tf.shape(Y)[0]) * 0.8) :],
        )

    # model
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            input_shape=input_shape[1:],
        )
    )
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(len(one_hot_map.keys())))

    model.summary()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_X,
        train_Y,
        epochs=10,
        validation_data=(val_X, val_Y),
    )

    test_loss, test_acc = model.evaluate(test_X, test_Y)

    print(test_loss, test_acc)
    pr.disable()
