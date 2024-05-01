"""
Example of a simple MNIST image classifier using the hyper-sinh activation function in its two
convolutional layers in Tf.keras.

Adapted from https://keras.io/examples/vision/mnist_convnet/
"""

import numpy as np
from src.examples.constants import (BATCH_SIZE, DROPOUT, IMAGE_DIM,
                                    KERNEL_SIZE_CONV, KERNEL_SIZE_MAX_POOL,
                                    NUM_CLASSES, NUM_EPOCHS, OUT_CHANNEL_CONV1,
                                    OUT_CHANNEL_CONV2)
from src.tf_keras.hyper_sinh import HyperSinh
from tensorflow import keras
from tensorflow.keras import layers

"""
## Prepare the data
"""

# Model / data parameters
inputs_shape = (IMAGE_DIM, IMAGE_DIM, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

"""
## Build the model
"""

model = keras.Sequential(
    [
        keras.Input(shape=inputs_shape),

        # First convolutional layer with the 'HyperSinh' activation function
        layers.Conv2D(OUT_CHANNEL_CONV1, kernel_size=(
            KERNEL_SIZE_CONV, KERNEL_SIZE_CONV)),
        # Instead of layers.Conv2D(64, kernel_size=(3, 3), activation="relu") when using the ReLU activation
        HyperSinh(),

        layers.MaxPooling2D(pool_size=(
            KERNEL_SIZE_MAX_POOL, KERNEL_SIZE_MAX_POOL)),

        # Second convolutional layer with the 'HyperSinh' activation function
        layers.Conv2D(OUT_CHANNEL_CONV2, kernel_size=(
            KERNEL_SIZE_CONV, KERNEL_SIZE_CONV)),
        HyperSinh(),

        layers.MaxPooling2D(pool_size=(
            KERNEL_SIZE_MAX_POOL, KERNEL_SIZE_MAX_POOL)),
        layers.Flatten(),
        layers.Dropout(DROPOUT),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model.summary()

"""
## Train the model
"""

model.compile(loss="categorical_crossentropy",
              optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=BATCH_SIZE,
          epochs=NUM_EPOCHS, validation_split=0.1)

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
