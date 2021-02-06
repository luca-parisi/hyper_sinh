# hyper-sinh as a custom activation function in TensorFlow

# Defining the hyper-sinh function

import numpy as np

def hyper_sinh(x):

  if x>0:
    x = 1/3*np.sinh(x)
    return x

  else:
    x = 1/4*(x**3)
    return x

# Vectorising the hyper-sinh function  
np_hyper_sinh = np.vectorize(hyper_sinh)

# Defining the derivative of the function hyper-sinh

def d_hyper_sinh(x):

  if x>0:
    x = 1/3*np.cosh(x)
    return x

  else:
    x = 3/4*(x**2)
    return x

np_d_hyper_sinh = np.vectorize(d_hyper_sinh)

# Defining the gradient function of the hyper-sinh

def hyper_sinh_grad(op, grad):
    x = op.inputs[0]
    n_gr = tf_d_hyper_sinh(x)
    return grad * n_gr

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
# Generating a unique name to avoid duplicates
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+2))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)
        
np_hyper_sinh_32 = lambda x: np_hyper_sinh(x).astype(np.float32)
def tf_hyper_sinh(x,name=None):
    with tf.name_scope(name, "hyper_sinh", [x]) as name:
        y = py_func(np_hyper_sinh_32,   #forward pass function
                        [x],
                        [tf.float32],
                        name=name,
                         grad= hyper_sinh_grad) # The function that overrides gradient
        y[0].set_shape(x.get_shape())     # Specify input rank
        return y[0]
np_d_hyper_sinh_32 = lambda x: np_d_hyper_sinh(x).astype(np.float32)
def tf_d_hyper_sinh(x,name=None):
    with tf.name_scope(name, "d_hyper_sinh", [x]) as name:
        y = tf.py_func(np_d_hyper_sinh_32,
                        [x],
                        [tf.float32],
                        name=name,
                        stateful=False)
        return y[0]

# Example of usage in TensorFlow with a hyper-sinh layer between a convolutional layer (#2) and a pooling layer (#2)
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same")
  conv2_act = tf_hyper_sinh(conv2)
  pool2 = tf.layers.max_pooling2d(inputs=conv2_act, pool_size=[2, 2], strides=2)

# hyper-sinh as a custom layer in Keras 
from tensorflow.keras.layers import Layer

class hyper_sinh(Layer):

    def __init__(self):
        super(hyper_sinh,self).__init__()

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs,name=None):
        return tf_hyper_sinh(inputs,name=None)

    def get_config(self):
        base_config = super(hyper_sinh, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

# Example of usage in a sequential model in Keras with a hyper-sinh layer between a convolutional layer and a pooling layer

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(hyper_sinh())
model.add(layers.MaxPooling2D((2, 2)))
