import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import util
import pandas as pd
from database import Database
import os


class Classifier:
    def __init__(self):
        # Add logic to load Hyperparameters here
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.session = tf.Session(config=self.config)
        self.model = None

    def create_model(self):
        self.model = tf.keras.Sequential([
            # Layer 1
            tf.keras.layers.Conv2D(input_shape=(300, 300, 3), filters=32, padding='same', kernel_size=[3, 3], strides=1,
                                   activation=tf.nn.elu, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                                   bias_initializer="zeros", kernel_regularizer=tf.keras.regularizers.l2(0.005)),
            tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2),
            tf.keras.layers.Dropout(0.3),

            # Layer 2
            tf.keras.layers.Conv2D(filters=32, padding='same', kernel_size=[3, 3], strides=1, activation=tf.nn.elu,
                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None), bias_initializer="zeros",
                                   kernel_regularizer=tf.keras.regularizers.l2(0.005)),
            tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2),
            tf.keras.layers.Dropout(0.3),

            # Layer 3
            tf.keras.layers.Conv2D(filters=64, padding='same', kernel_size=[3, 3], strides=1, activation=tf.nn.elu,
                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None), bias_initializer="zeros",
                                   kernel_regularizer=tf.keras.regularizers.l2(0.005)),
            tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2),
            tf.keras.layers.Dropout(0.3),

            # Layer 4
            tf.keras.layers.Conv2D(filters=64, padding='same', kernel_size=[3, 3], strides=1, activation=tf.nn.elu,
                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None), bias_initializer="zeros",
                                   kernel_regularizer=tf.keras.regularizers.l2(0.005)),
            tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2),
            tf.keras.layers.Dropout(0.3),

            # Layer 5
            tf.keras.layers.Conv2D(filters=128, padding='same', kernel_size=[3, 3], strides=1, activation=tf.nn.elu,
                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None), bias_initializer="zeros",
                                   kernel_regularizer=tf.keras.regularizers.l2(0.005)),
            tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2),
            tf.keras.layers.Dropout(0.3),

            # Layer 6
            tf.keras.layers.Conv2D(filters=128, padding='same', kernel_size=[3, 3], strides=1, activation=tf.nn.elu,
                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None), bias_initializer="zeros",
                                   kernel_regularizer=tf.keras.regularizers.l2(0.005)),
            tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2),
            tf.keras.layers.Dropout(0.3),

            # Layer 7
            tf.keras.layers.Conv2D(filters=128, padding='same', kernel_size=[3, 3], strides=1, activation=tf.nn.elu,
                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None), bias_initializer="zeros",
                                   kernel_regularizer=tf.keras.regularizers.l2(0.005)),
            tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2),
            tf.keras.layers.Dropout(0.3),

            # Layer 8
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None), bias_initializer="zeros",
                            kernel_regularizer=tf.keras.regularizers.l2(0.005)),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(1024, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None), bias_initializer="zeros",
                            kernel_regularizer=tf.keras.regularizers.l2(0.005)),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Dense(825, activation=None, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                            bias_initializer="zeros", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Activation('softmax')
        ])

        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.00001, decay=10e-7),
                           loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"])

        self.model.summary()

    def load_model(self):
        self.create_model()
        if os.path.exists('./models/model.h5'):
            self.model.load_weights('./models/model.h5')

    def train(self, x_train, y_train, epochs=1, batch_size=32):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle="batch",)
        self.model.save_weights('./models/model.h5')

    def evaluate(self, x_test, y_test):
        self.model.evaluate(x_test, y_test)

    def predict(self, x_test):
        return self.model.predict(x_test)


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    classifier = Classifier()
    classifier.load_model()

    hf = h5py.File('x_train_1.h5', 'r')
    x_train_1 = hf.get('x_train_1')
    x_train_1 = np.array(x_train_1, dtype=np.float32) / 255

    hf = h5py.File('y_train_1.h5', 'r')
    y_train_1 = hf.get('y_train_1')
    y_train_1 = np.array(y_train_1)

    hf = h5py.File('x_train_2.h5', 'r')
    x_train_2 = hf.get('x_train_2')
    x_train_2 = np.array(x_train_2, dtype=np.float32) / 255

    hf = h5py.File('y_train_2.h5', 'r')
    y_train_2 = hf.get('y_train_2')
    y_train_2 = np.array(y_train_2)

    hf = h5py.File('x_test.h5', 'r')
    x_test = hf.get('x_test')
    x_test = np.array(x_test, dtype=np.float32) / 255

    hf = h5py.File('y_test.h5', 'r')
    y_test = hf.get('y_test')
    y_test = np.array(y_test)

    for i in range(100):
        print("EPOCH", i)

        classifier.train(x_train_1, y_train_1, epochs=1, batch_size=100)
        classifier.train(x_train_2, y_train_2, epochs=1, batch_size=100)

        if i % 10 == 0:
            print("Evaluation")
            classifier.evaluate(x_test, y_test)
