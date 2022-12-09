from genericpath import exists
import os
from collections import deque
from keras.saving.legacy.save import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from tensorflow.python.ops.gen_array_ops import one_hot
from tqdm import tqdm
import numpy as np
import tensorflow as tf

dev = 1

DATASET_PATH = "./ds/"

def get_data_from_dir(path: str) -> tuple[tuple[deque, deque], dict[str, int]]:
    """Function for getting and preparing data

    This function will return data and corrosponding labels, along
    with a map of classes to get string labels.

    Args:
        path (str): Path to directory with .npy files.

    Returns:
        tuple[tuple[deque, deque], dict[str, int]]: A tuple with tuple of data and label,
        and a map for classes.
    """
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
            x = 0
            if dev == 1:
                x = np.load(root + "/" + file, allow_pickle=True)[:num_samples]
            else:
                x = np.load(root + "/" + file, allow_pickle=True)
            X.extend(tf.reshape(x, shape=(x.shape[0], 28, 28, 1)))
            Y.extend([k for k, v in class_map.items() if v == key] * len(x))
            count += 1

    return ((X, Y), class_map)


if __name__ == "__main__":
    # Getting data
    ((X, Y), class_map) = get_data_from_dir(DATASET_PATH)

    # Converting data and lables to tensors 
    # Converting from np.array is better/performant than from list
    X, Y = tf.convert_to_tensor(np.array(X), dtype=tf.float32), tf.convert_to_tensor(np.array(Y), dtype=tf.int32)

    # Shuffling the data as classes are contiguous
    indices = tf.range(start=0, limit=tf.shape(X)[0])
    shuffled_indices = tf.random.shuffle(indices)
    X, Y = tf.gather(X, shuffled_indices), tf.gather(Y, shuffled_indices)

    # Splitting data for training and testing with 20% used as testing set and 20% of the training set used as validation set
    if dev == 1:
        k = 10_000
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

    # Defining CNN with 2 sets of Conv2D and Maxpooling2D layers, followed by a Flattening layer, Dense layer, Dropout layer, and another Dense layer
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation=tf.nn.relu, kernel_initializer="he_normal"))
    model.add(tf.keras.layers.MaxPooling2D(strides=1, pool_size=2))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=1, activation=tf.nn.relu, kernel_initializer="he_normal"))
    model.add(tf.keras.layers.MaxPooling2D(strides=1, pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=1024, activation=tf.nn.relu, kernel_initializer="he_normal"))
    model.add(tf.keras.layers.Dropout(rate=5e-1))
    model.add(tf.keras.layers.Dense(units=len(class_map.keys())))

    # If model is already saved, load model.
    if exists("./checkpoints/model"):
        model = load_model("./checkpoints/model")
    # Else compile, fit and save the model
    else:
        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        history = model.fit(
            train_X,
            train_Y,
            epochs=16,
            validation_data=(val_X, val_Y),
        )
        model.save("./checkpoints/model")

    # Printing model summary
    model.summary()

    # Converting int labels to string labels
    test_labels = []
    for t in test_Y:
        test_labels.append(class_map[int(t)])

    # Generate prediction using test data
    model_preds = model.predict(test_X.numpy(), verbose='0')

    # Converting prediction values to string labels 
    pred_labels = []
    for p in (pb := tqdm(model_preds)):
        pb.set_description("converting predictions")
        pred_labels.append(class_map[np.argmax(p)])

    # One Hot Encoding test_Y for AUC for ROC
    one_hot_test_Y = tf.one_hot(test_Y, depth=len(class_map.keys()))

    # Generating and printing accuracy, AUC for ROC, and confusion matrix
    print(f"accuracy_score: \n{accuracy_score(test_labels, pred_labels)}")
    print(f"\nroc_auc_score: \n{roc_auc_score(np.array(one_hot_test_Y), model_preds)}")
    print(f"\nconfusion_matrix: \n{confusion_matrix(test_labels, pred_labels)}")

    # write params to file
    # if True:
    #     with open("./params2.csv", "a") as file:
    #         file.write(
    #             f"{c1_filters}, {c1_kernel_size}, {c1_strides}, {c1_activation.__name__}, {m1_strides}, {m1_pool_size}, {c2_filters}, {c2_kernel_size}, {c2_strides}, {c2_activation.__name__}, {m1_strides}, {m2_pool_size}, {d1_units}, {do_rate}, {d2_units}, {kernel_initializer}, {lr}, {epochs}, {test_acc}, {test_loss}\n"
    #         )
