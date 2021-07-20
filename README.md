# hyper-sinh in TensorFlow and Keras
## An Accurate and Reliable Function from Shallow to Deep Learning

The **'hyper-sinh'** is a Python custom activation function available for both shallow and deep neural networks in TensorFlow and Keras for Machine Learning- and Deep Learning-based classification. It is distributed under the [CC BY 4.0 license](http://creativecommons.org/licenses/by/4.0/).

Details on this function, implementation and validation against gold standard activation functions for both shallow and deep neural networks are available at the following paper: **[Parisi et al., 2021a](https://www.sciencedirect.com/science/article/pii/S2666827021000566)**. 


### Dependencies

Developed in Python 3.6, as compatible with TensorFlow (versions tested: 1.12 and 1.15) and Keras, please note the dependencies of TensorFlow (v1.12 or 1.15) and Keras to be able to use the 'hyper-sinh' activation function in shallow and deep neural networks.


### Usage

You can use the custom hyper-sinh activation function in Keras as a layer:

#### Example of usage in a sequential model in Keras with a hyper-sinh layer between a convolutional layer and a pooling layer

```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(hyper_sinh()) 
model.add(layers.MaxPooling2D((2, 2)))
```

### Citation request

If you are using this function, please cite the papers by:
* **[Parisi et al., 2020](https://arxiv.org/abs/2011.07661)**.
* **[Parisi et al., 2021a](https://www.sciencedirect.com/science/article/pii/S2666827021000566)**.
* **[Parisi et al., 2021b](https://www.wseas.org/multimedia/journals/computerresearch/2021/a025118-001(2021).pdf)**.
