import os
from typing import Final

import matplotlib.pyplot as plt
import tensorflow as tf
from loguru import logger
from tensorflow.python.keras import Input, Model, Sequential
from tensorflow.python.keras import layers as L
from tensorflow.python.keras.callbacks import EarlyStopping, History

_MODEL_OUTPUT_PATH: Final[str] = "cnn_model.h5"
_BATCH_SIZE: Final[int] = 20
_EPOCHS: Final[int] = 1
_LOSS: Final[str] = "mae"
_CHANNELS: Final[int] = 4
_VALIDATION_SPLIT: Final[float] = 0.3


def make_model(*, input_shape=(64, 64, _CHANNELS), expo=6, dropout=0.0):
    def conv_block(
        name, filters, kernel_size=4, pad="same", t=False, act="relu", bn=True
    ):
        block = Sequential(name=name)

        if act == "relu":
            block.add(L.ReLU())
        elif act == "leaky_relu":
            block.add(L.LeakyReLU(0.2))

        if not t:
            block.add(
                L.Conv2D(
                    filters,
                    kernel_size=kernel_size,
                    strides=(2, 2),
                    padding=pad,
                    use_bias=True,
                    activation=None,
                )
            )
        else:
            block.add(L.UpSampling2D(interpolation="bilinear"))
            block.add(
                L.Conv2DTranspose(
                    filters=filters,
                    kernel_size=kernel_size - 1,
                    padding=pad,
                    activation=None,
                )
            )

        if dropout > 0:
            block.add(L.SpatialDropout2D(dropout))

        # if bn:
        #     block.add(L.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9))

        return block

    channels = int(2**expo + 0.5)
    e0 = Sequential(name="enc_0")
    e0.add(
        L.Conv2D(
            filters=_CHANNELS,
            kernel_size=4,
            strides=(2, 2),
            padding="same",
            activation=None,
            data_format="channels_last",
        )
    )
    e1 = conv_block("enc_1", channels, act="leaky_relu")
    e2 = conv_block("enc_2", channels * 2, act="leaky_relu")
    e3 = conv_block("enc_3", channels * 4, act="leaky_relu")
    e4 = conv_block("enc_4", channels * 8, act="leaky_relu", kernel_size=2, pad="valid")
    e5 = conv_block("enc_5", channels * 8, act="leaky_relu", kernel_size=2, pad="valid")

    dec_5 = conv_block("dec_5", channels * 8, t=True, kernel_size=2, pad="valid")
    dec_4 = conv_block("dec_4", channels * 8, t=True, kernel_size=2, pad="valid")
    dec_3 = conv_block("dec_3", channels * 4, t=True)
    dec_2 = conv_block("dec_2", channels * 2, t=True)
    dec_1 = conv_block("dec_1", channels, act=None, t=True, bn=False)
    dec_0 = Sequential(name="dec_0")
    dec_0.add(L.ReLU())
    dec_0.add(L.Conv2DTranspose(_CHANNELS, kernel_size=4, strides=(2, 2), padding="same"))

    # Forward Pass
    inputs = Input(shape=input_shape)
    out0 = e0(inputs)
    out1 = e1(out0)
    out2 = e2(out1)
    out3 = e3(out2)
    out4 = e4(out3)
    out5 = e5(out4)

    dout5 = dec_5(out5)
    dout5_out4 = tf.concat([dout5, out4], axis=3)
    dout4 = dec_4(dout5_out4)
    dout4_out3 = tf.concat([dout4, out3], axis=3)
    dout3 = dec_3(dout4_out3)
    dout3_out2 = tf.concat([dout3, out2], axis=3)
    dout2 = dec_2(dout3_out2)
    dout2_out1 = tf.concat([dout2, out1], axis=3)
    dout1 = dec_1(dout2_out1)
    dout1_out0 = tf.concat([dout1, out0], axis=3)
    dout0 = dec_0(dout1_out0)

    model = Model(inputs=inputs, outputs=dout0)
    model.compile(optimizer="adam", loss=_LOSS)
    return model


def train_model(model: Model, x, y):
    history = model.fit(
        x,
        y,
        epochs=_EPOCHS,
        batch_size=_BATCH_SIZE,
        validation_split=_VALIDATION_SPLIT,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    )
    return history


def plot(history: History):
    logger.info("Plotting")
    history_dict = history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]

    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training & Validation Loss", fontsize=16)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend()

    plt.show()


def save_model(model):
    dirname = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    model.save(f"{dirname}/{_MODEL_OUTPUT_PATH}")
    logger.success("Model saved successfully")

    logger.info("Plotting")
