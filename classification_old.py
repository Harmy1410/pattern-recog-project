from genericpath import exists
import os
from collections import deque
from keras.saving.legacy.save import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tqdm import tqdm
import numpy as np
import tensorflow as tf

dev = 1

if dev == 0:
    DATASET_PATH = "./dataset/"
else:
    DATASET_PATH = "./ds/"


def get_data_from_dir(
    path: str,
) -> tuple[tuple[deque, deque], dict[str, int]]:
    num_samples = 20_000
    X = deque()
    Y = deque()
    class_map = {}
    count = 0
    for root, _, files in os.walk(path):
        for file in (pb := tqdm(files)):
            pb.set_description(f"Reading Files")
            if "DS_Store" in file:
                continue
            key = file[18:-4].lower().replace(" ", "_").replace("-", "_")
            class_map[count] = key
            x = np.load(root + "/" + file, allow_pickle=True)[:num_samples]
            X.extend(tf.reshape(x, shape=(x.shape[0], 28, 28, 1)))
            Y.extend([k for k, v in class_map.items() if v == key] * len(x))
            count += 1

    return ((X, Y), class_map)


if __name__ == "__main__":

    # getting data
    ((X, Y), class_map) = get_data_from_dir(DATASET_PATH)

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
        train_X, train_Y = (X[: int(int(int(tf.shape(X)[0]) * 0.8) * 0.8)], Y[: int(int(int(tf.shape(Y)[0]) * 0.8) * 0.8)])

        val_X, val_Y = (
            X[int(int(int(tf.shape(X)[0]) * 0.8) * 0.8) : int(int(tf.shape(X)[0]) * 0.8)],
            Y[int(int(int(tf.shape(Y)[0]) * 0.8) * 0.8) : int(int(tf.shape(X)[0]) * 0.8)]
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
    m1_strides = 1
    m1_pool_size = 2

    # c2_filters = [32, 64, 32, 64, 64, 32, 64, 32]
    # c2_kernel_size = [5, 3, 3, 5, 5, 5, 3, 3]
    c2_filters = 32
    c2_kernel_size = 3
    c2_strides = 1
    c2_activation = tf.nn.relu
    m2_strides = 1
    m2_pool_size = 2

    d1_units = 1024
    do_rate = 5e-1
    d2_units = len(class_map.keys())

    kernel_initializer = "he_normal"

    lr = 0.001
    epochs = 16

    # model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=c1_filters,kernel_size=c1_kernel_size,strides=c1_strides,activation=c1_activation,kernel_initializer=kernel_initializer))
    model.add(tf.keras.layers.MaxPooling2D(strides=m1_strides, pool_size=m1_pool_size))
    model.add(tf.keras.layers.Conv2D(filters=c2_filters,kernel_size=c2_kernel_size,strides=c2_strides,activation=c2_activation,kernel_initializer=kernel_initializer))
    model.add(tf.keras.layers.MaxPooling2D(strides=m2_strides, pool_size=m2_pool_size))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=d1_units,activation=tf.nn.relu,kernel_initializer=kernel_initializer))
    model.add(tf.keras.layers.Dropout(rate=do_rate))
    model.add(tf.keras.layers.Dense(units=d2_units))

    if exists("./checkpoints/model"):
        model = load_model("./checkpoints/model")

    else:
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
        model.save_weights("./checkpoints/weights")
        model.save("./checkpoints/model")

    model.summary()
    test_loss, test_acc = model.evaluate(test_X, test_Y)

    print(f"\ntest_loss: {test_loss}, test_acc: {test_acc}")

    pred_classes = []

    for t in test_X:
        pred_classes.append(model.predict(np.array([t])))

    char_test_labels = class_map[pred.tolist().index(max(pred))]
    char_model_predictions = []

    print(accuracy_score(char_test_labels, char_model_predictions))
    print(roc_auc_score(np.array(test_Y), model_predictions))
    confusion_matrix(char_test_labels, char_model_predictions)

    # write params to file
    if True:
        with open("./params2.csv", "a") as file:
            file.write(
                f"{c1_filters}, {c1_kernel_size}, {c1_strides}, {c1_activation.__name__}, {m1_strides}, {m1_pool_size}, {c2_filters}, {c2_kernel_size}, {c2_strides}, {c2_activation.__name__}, {m1_strides}, {m2_pool_size}, {d1_units}, {do_rate}, {d2_units}, {kernel_initializer}, {lr}, {epochs}, {test_acc}, {test_loss}\n"
            )
