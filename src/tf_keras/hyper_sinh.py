"""Additional utility for Machine Learning and Deep Learning models in TensorFlow and Keras"""

# The hyperbolic sinh or 'hyper-sinh' as a custom activation function in TensorFlow (tf_hyper_sinh)
# and Keras (HyperSinh)

# Author: Luca Parisi <luca.parisi@ieee.org>

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Layer
from tensorflow.lite.python.op_hint import _LiteFuncCall

from tf_keras.utils import py_func

from .constants import (GT_COEFFICIENT, LE_DERIV_FIRST_COEFF,
                        LE_DERIV_SECOND_COEFF, LE_FIRST_COEFFICIENT,
                        LE_SECOND_COEFFICIENT, NAME_DERIV_HYPER_SINH,
                        NAME_HYPER_SINH)

# hyper-sinh as a custom activation function in TensorFlow


'''
# Example of usage of the hyper-sinh in TensorFlow as a custom activation function of a convolutional layer (#2)

convolutional_layer_2 = tf.layers.conv2d(
                        inputs=pooling_layer_1,
                        filters=64,
                        kernel_size=[5, 5],
                        padding="same")
convolutional_layer_activation = tf_hyper_sinh(convolutional_layer_2)
pooling_layer_2 = tf.layers.max_pooling2d(inputs=convolutional_layer_activation, pool_size=[2, 2], strides=2)
'''


# Defining the hyper-sinh function
def hyper_sinh(x: float) -> float:
    """
    Apply the hyper-sinh activation function to transform inputs accordingly.

    Args:
        x: float
            The input to be transformed via the hyper-sinh activation function.

    Returns:
            The transformed x (float) via the hyper-sinh.
    """

    if x > 0:
        x = GT_COEFFICIENT * np.sinh(x)
        return x
    else:
        x = LE_FIRST_COEFFICIENT * (x**LE_SECOND_COEFFICIENT)
        return x


# Vectorising the 'hyper_sinh' function
np_hyper_sinh = np.vectorize(hyper_sinh)


def derivative_hyper_sinh(x: float) -> float:
    """
    Compute the derivative of the hyper-sinh activation function.

    Args:
        x: float
            The input from which the derivative is to be computed.

    Returns:
            The derivative (float) of the hyper-sinh given an input.
    """

    if x > 0:
        x = GT_COEFFICIENT * np.cosh(x)
        return x
    else:
        x = LE_DERIV_FIRST_COEFF * (x**LE_DERIV_SECOND_COEFF)
        return x


# Vectorising the derivative of the hyper-sinh function
np_der_hyper_sinh = np.vectorize(derivative_hyper_sinh)


def hyper_sinh_grad(op: _LiteFuncCall, grad: float) -> float:
    """
    Define the gradient function of the hyper-sinh.

    Args:
        op: _LiteFuncCall
            A TensorFlow Lite custom function.
        grad:
            The input gradient.

    Returns:
            The gradient function of the hyper-sinh.
    """

    x = op.inputs[0]
    n_gr = tf_der_hyper_sinh(x)
    return grad * n_gr


def np_hyper_sinh_float32(x): return np_hyper_sinh(x).astype(np.float32)


def tf_hyper_sinh(x):
    """
    The hyper-sinh activation function defined in TensorFlow.

    Args:
        x: Tensor
            The input tensor.

    Returns:
            The output tensor (Tensor) from the hyper-sinh activation function.
    """

    name = NAME_HYPER_SINH

    y = py_func(
        np_hyper_sinh_float32,  # Forward pass function
        [x],
        [tf.float32],
        name=name,
        grad=hyper_sinh_grad
    )  # The function that overrides gradient
    y[0].set_shape(x.get_shape())  # To specify the rank of the input
    return y[0]


def np_der_hyper_sinh_float32(
    x): return np_der_hyper_sinh(x).astype(np.float32)


def tf_der_hyper_sinh(x: list[Tensor]) -> float:
    """
    The derivative of the hyper-sinh defined in TensorFlow.

    Args:
        x: list[Tensor]
            A list of input tensors.

    Returns:
            The output computed as the derivative of the hyper-sinh activation function.
    """

    name = NAME_DERIV_HYPER_SINH

    y = py_func(
        np_der_hyper_sinh_float32,
        [x],
        [tf.float32],
        name=name,
        stateful=False
    )
    return y[0]


# HyperSinh as a custom layer in Keras

'''
Either

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(HyperSinh()) 
model.add(layers.MaxPooling2D((2, 2)))

or

model = keras.Sequential(
        keras.Input(shape=(32, 32, 3)),

        layers.Conv2D(32, kernel_size=(3, 3)),
        HyperSinh(),

        layers.MaxPooling2D(pool_size=(2, 2)),
    ]
)
'''


class HyperSinh(Layer):
    """
    A class defining the HyperSinh activation function in keras.
    """

    def __init__(self) -> None:
        """
        Initialise the HyperSinh activation function.
        """

        super().__init__()

    def build(self, input_shape: tuple[int, int, int]) -> None:
        """
        Build the HyperSinh activation function given an input shape.

        Args:
            input_shape: tuple[int, int, int]
                        The shape of the input tensor considered.
        """

        super().build(input_shape)

    def call(self, inputs: Tensor) -> Tensor:
        """
        Call the HyperSinh activation function.

        Args:
            inputs: Tensor
                    The input tensor.

        Returns:
                The output tensor (Tensor) from the HyperSinh activation function.
        """

        return tf_hyper_sinh(inputs)

    def get_config(self):
        """
        Get the configs of the HyperSinh activation function.

        Returns:
                A dictionary of the configs of the HyperSinh activation function.
        """

        base_config = super().get_config()
        return dict(list(base_config.items()))
