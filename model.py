import tensorflow as tf
from tensorflow.keras import layers as L, models as M

BATCHSZIE = 8
S = 96
AUTOTUNE = tf.data.experimental.AUTOTUNE


def down(filters, kernel, pool):
    m = M.Sequential()
    m.add(L.Conv2D(filters, kernel, padding="same", activation="elu"))
    m.add(L.BatchNormalization())
    m.add(L.MaxPooling2D(pool_size=pool, strides=pool))
    return m


def model(w, c):
    bitmap = L.Input(shape=[S, S, 1])
    common = bitmap
    stack = [
        down(32, (6, 6), (6, 6)),
        down(64, (5, 5), (4, 4)),
        down(128, (5, 5), (2, 2)),
        down(256, (5, 5), (2, 2)),
        # down(512, (5, 5), (2, 2))
    ]
    pool = [(16, 16), (4, 4), (2, 2), (1, 1)]
    cc = []
    N = 0
    for p in stack:
        common = p(common)
        cc.append(L.MaxPooling2D(pool_size=pool[N], strides=pool[N])(common))
        N = N + 1
    for m in cc[:-1]:
        common = L.Concatenate()([common, m])
    common = L.Flatten()(common)
    wr = L.Dense(w, activation="softmax", name="writer")(common)
    ch = L.Dense(c, activation="softmax", name="character")(common)
    return tf.keras.Model(inputs=bitmap, outputs=[wr, ch])


if __name__ == "__main__":
    w = 60
    c = 10000
    m = model(w, c)
    m.summary()
    m.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
