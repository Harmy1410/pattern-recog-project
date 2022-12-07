import os
from collections import deque
import cProfile
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
import tensorflow as tf

dev = 1

if dev == 0:
    DATASET_PATH = "./dataset/"
else:
    DATASET_PATH = "./ds/"


def get_data_from_dir(
    path: str,
) -> tuple[tuple[deque, deque], dict[str, int]]:
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

    X = deque()
    Y = deque()
    for key, items in data.items():
        arr = items[:5000]
        X.extend(arr)
        Y.extend([class_map[key]] * len(arr))

    return ((X, Y), class_map)


if __name__ == "__main__":

    # pr = cProfile.Profile()
    # pr.enable()
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
        k = 6400
        j = int(k * 0.8)
        i = int(j * 0.8)
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

    # c1_filters = [64, 32, 64, 32, 64, 32, 64, 32]
    # c1_kernel_size = [3, 5, 5, 3, 5, 5, 3, 3]
    c1_filters = 32
    c1_kernel_size = 3
    c1_strides = 1
    c1_activation = tf.nn.relu
    m1_strides = [1, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1]
    m1_pool_size = [2, 1, 2, 1, 1, 2, 1, 2, 1, 2, 2, 1]

    # c2_filters = [32, 64, 32, 64, 64, 32, 64, 32]
    # c2_kernel_size = [5, 3, 3, 5, 5, 5, 3, 3]
    c2_filters = 32
    c2_kernel_size = 3
    c2_strides = 1
    c2_activation = tf.nn.relu
    m2_strides = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    m2_pool_size = [2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2]

    d1_units = 1024
    do_rate = 5e-1
    d2_units = len(one_hot_map.keys())

    kernel_initializer = "he_normal"

    lr = 0.002
    epochs = 16

    # model
    for i in range(len(m1_pool_size)):
        print(f" ---------- {i} ---------- ")
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Conv2D(
                filters=c1_filters,
                kernel_size=c1_kernel_size,
                strides=c1_strides,
                activation=c1_activation,
                kernel_initializer=kernel_initializer,
            )
        )
        model.add(
            tf.keras.layers.MaxPooling2D(
                strides=m1_strides[i], pool_size=m1_pool_size[i]
            )
        )
        model.add(
            tf.keras.layers.Conv2D(
                filters=c2_filters,
                kernel_size=c2_kernel_size,
                strides=c2_strides,
                activation=c2_activation,
                kernel_initializer=kernel_initializer,
            )
        )
        model.add(
            tf.keras.layers.MaxPooling2D(
                strides=m2_strides[i], pool_size=m2_pool_size[i]
            )
        )
        model.add(tf.keras.layers.Flatten())
        model.add(
            tf.keras.layers.Dense(
                units=d1_units,
                activation=tf.nn.relu,
                kernel_initializer=kernel_initializer,
            )
        )
        model.add(tf.keras.layers.Dropout(rate=do_rate))
        model.add(tf.keras.layers.Dense(units=d2_units))

        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=lr),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        history = model.fit(
            train_X,
            train_Y,
            epochs=epochs,
            validation_data=(val_X, val_Y),
        )
        model.summary()

        test_loss, test_acc = model.evaluate(test_X, test_Y)

        print(f"\ntest_loss: {test_loss}, test_acc: {test_acc}")

        # write params to file
        if True:
            with open("./params2.csv", "a") as file:
                file.write(
                    f"{c1_filters}, {c1_kernel_size}, {c1_strides}, {c1_activation.__name__}, {m1_strides[i]}, {m1_pool_size[i]}, {c2_filters}, {c2_kernel_size}, {c2_strides}, {c2_activation.__name__}, {m1_strides[i]}, {m2_pool_size[i]}, {d1_units}, {do_rate}, {d2_units}, {kernel_initializer}, {lr}, {epochs}, {test_acc}, {test_loss}\n"
                )
    # pr.disable()
