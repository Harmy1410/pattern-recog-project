from genericpath import exists
import os
from os import listdir
from collections import deque
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
    Conv2DTranspose,
    BatchNormalization
)
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.callbacks import EarlyStopping


def load_data(root: str = "", int_label=True) -> tuple:
    labels = []
    data = None
    files = [dir for dir in listdir(root) if ".npy" in dir]
    i = 0
    for dir in files:
        label = dir.split(".")[0]
        temp_data = np.load(root + dir)
        hot_encoded_label = [0] * len(files)
        hot_encoded_label[i] = 1
        labels += (
            [hot_encoded_label] * temp_data.shape[0]
            if int_label
            else [label] * temp_data.shape[0]
        )
        data = temp_data if data is None else np.append(data, temp_data, 0)
        i += 1
    # The original data are 28, 28 grayscale bitmaps
    return data.reshape(-1, 28, 28, 1).astype("float") / np.max(data), labels


def get_autoencoder(train_data, test_data) -> tuple:
    if exists("./content/autoencoder"):
        model = load_model("./content/autoencoder")
        encoder = Model(model.input, model.layers[2].output)
        decoder_input = Input(shape=(7, 7, 3))
        decoder_layer1 = model.layers[-2](decoder_input)
        decoder = Model(decoder_input, model.layers[-1](decoder_layer1))
        return encoder, decoder

    input = Input(shape=(28, 28, 1))
    layer = Conv2D(32, (4, 4), activation="relu", strides=(2, 2), padding="same")(input)
    layer = Conv2D(3, (2, 2), activation="sigmoid", strides=(2, 2), padding="same")(
        layer
    )
    layer = Conv2DTranspose(
        32, (2, 2), activation="relu", strides=(2, 2), padding="same"
    )(layer)
    layer = Conv2DTranspose(
        1, (4, 4), activation="sigmoid", strides=(2, 2), padding="same"
    )(layer)
    model = Model(input, layer)
    model.compile(optimizer="adam", loss="binary_crossentropy")

    callback = EarlyStopping(monitor="loss", patience=3)
    model.fit(
        train_data,
        train_data,
        epochs=16,
        batch_size=128,
        shuffle=True,
        validation_data=(test_data, test_data),
        callbacks=[callback],
    )
    model.save("./content/autoencoder")

    encoder = Model(model.input, model.layers[2].output)
    decoder_input = Input(shape=(7, 7, 3))
    decoder_layer1 = model.layers[-2](decoder_input)
    decoder = Model(decoder_input, model.layers[-1](decoder_layer1))
    return encoder, decoder


if __name__ == "__main__":

    train_data, test_data, train_labels, test_labels = train_test_split(
        *load_data("./ds/"), test_size=0.2
    )
    encoder, decoder = get_autoencoder(train_data, test_data)

    train_data, valid_data, train_labels, valid_labels = train_test_split(
        train_data, train_labels, test_size=0.2
    )

    encoded_train_data = encoder.predict(train_data)
    encoded_test_data = encoder.predict(test_data)
    encoded_valid_data = encoder.predict(valid_data)

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    valid_labels = np.array(valid_labels)

    # model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=2,strides=1,activation=tf.nn.relu,kernel_initializer="he_normal"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=16,kernel_size=2,strides=1,activation=tf.nn.relu,kernel_initializer="he_normal"))
    model.add(BatchNormalization(momentum=0.99))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=550, activation=tf.nn.relu))
    model.add(Dropout(rate=4e-1))
    model.add(Dense(units=5))
    if exists("./checkpoints/encodercnn"):
        model = load_model("./checkpoints/encodercnn")

    else:
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),loss="binary_crossentropy", metrics=["accuracy"])
        model.build(encoded_train_data.shape)
        model.summary()

        history = model.fit(encoded_train_data,train_labels,epochs=16,validation_data=(encoded_valid_data, valid_labels))
        model.save("./checkpoints/encodercnn")

    test_loss, test_acc = model.evaluate(encoded_test_data, test_labels)

    char_labels = ['car', 'bird', 'airplane', 'truck', 'ship']

    char_test_labels = []
    for label in test_labels:
      char_test_labels.append(char_labels[np.where(label == 1)[0][0]])

    model_predictions = model.predict(encoded_test_data)
    char_model_predictions = []

    for label in model_predictions:
      char_model_predictions.append(char_labels[np.argmax(label)])

    print(f"accuracy_score: \n{accuracy_score(char_test_labels, char_model_predictions)}")
    print(f"\nroc_auc_score: \n{roc_auc_score(np.array(test_labels), model_predictions)}")
    print(f"\nconfusion_matrix: \n{confusion_matrix(char_test_labels, char_model_predictions)}")
