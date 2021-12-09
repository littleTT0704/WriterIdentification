import os
import numpy as np, tensorflow as tf
from matplotlib import pyplot
from tensorflow.keras import layers as L, models as M
import struct
import random
import cv2

pyplot.rcParams["font.sans-serif"] = ["SimHei"]
pyplot.rcParams["axes.unicode_minus"] = False

BATCHSZIE = 8
S = 96
AUTOTUNE = tf.data.experimental.AUTOTUNE


def down(filters, kernel, pool):
    m = M.Sequential()
    m.add(L.Conv2D(filters, kernel, padding="same", activation="relu"))
    m.add(L.BatchNormalization())
    m.add(L.MaxPooling2D(pool_size=pool, strides=pool))
    return m


def model(w, c):
    bitmap = L.Input(shape=[S, S, 1])
    wr = ch = bitmap
    wr_stack = [
        down(32, (6, 6), (6, 6)),
        down(64, (5, 5), (4, 4)),
        down(128, (5, 5), (2, 2)),
        # down(256, (5, 5), (2, 2))
        # down(512, (5, 5), (2, 2))
    ]
    wr_pool = [
        # (8, 8),
        (8, 8),
        (2, 2),
        (1, 1),
    ]
    ch_stack = [
        down(32, (6, 6), (6, 6)),
        down(64, (5, 5), (4, 4)),
        down(128, (5, 5), (2, 2)),
        # down(256, (5, 5), (2, 2))
        # down(512, (5, 5), (2, 2))
    ]
    ch_pool = [
        # (8, 8),
        (8, 8),
        (2, 2),
        (1, 1),
    ]
    cc = []
    N = 0
    for p in wr_stack:
        wr = p(wr)
        cc.append(L.MaxPooling2D(pool_size=wr_pool[N], strides=wr_pool[N])(wr))
        N = N + 1
    for m in cc[:-1]:
        wr = L.Concatenate()([wr, m])
    vc = L.Flatten()(wr)
    wr = L.Dense(w, activation="softmax", name="writer")(vc)
    cc = []
    N = 0
    for p in ch_stack:
        ch = p(ch)
        cc.append(L.MaxPooling2D(pool_size=wr_pool[N], strides=wr_pool[N])(ch))
        N = N + 1
    for m in cc[:-1]:
        ch = L.Concatenate()([ch, m])
    ch = L.Flatten()(ch)
    ch = L.Dense(c, activation="softmax", name="character")(ch)
    return tf.keras.Model(inputs=bitmap, outputs=[wr, ch])


if __name__ == "__main__":
    w = 60
    c = 10000
    m = model(w, c)
    m.summary()
    m.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    m.load_weights("../v1/10.h5")
