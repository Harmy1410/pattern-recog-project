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
    X, Y = tf.convert_to_tensor(np.array(X), dtype=tf.float32), tf.convert_to_tensor(
        np.array(Y), dtype=tf.float32
    )

    # shuffling
    indices = tf.range(start=0, limit=tf.shape(X)[0])
    shuffled_indices = tf.random.shuffle(indices)
    X, Y = tf.gather(X, shuffled_indices), tf.gather(Y, shuffled_indices)
    input_shape = (tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2], 1)

    train_X, train_Y, val_X, val_Y, test_X, test_Y = 0, 0, 0, 0, 0, 0

    # splitting data
    if dev == 1:
        i = 5120
        j = 6400
        k = 8000
        train_X, train_Y = (X[:i], Y[:i])
        val_X, val_Y = (X[i:j], Y[i:j])
        test_X, test_Y = (X[j:k], Y[j:k])
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
            filters=32,
            kernel_size=5,
            strides=1,
            activation=tf.nn.relu,
            kernel_initializer="he_normal",
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(strides=1, pool_size=2))
    model.add(
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=5,
            strides=1,
            activation=tf.nn.relu,
            kernel_initializer="he_normal",
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(strides=1, pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(
        tf.keras.layers.Dense(
            units=1024, activation=tf.nn.relu, kernel_initializer="he_normal"
        )
    )
    model.add(tf.keras.layers.Dropout(rate=5e-1))
    model.add(tf.keras.layers.Dense(units=len(one_hot_map.keys())))

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=1e-3),
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(
        train_X,
        train_Y,
        epochs=10,
        validation_data=(val_X, val_Y),
    )
    model.summary()

    test_loss, test_acc = model.evaluate(test_X, test_Y)

    print(test_loss, test_acc)
    pr.disable()
