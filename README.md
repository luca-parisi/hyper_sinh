# hyper-sinh in TensorFlow and Keras
## An Accurate and Reliable Function from Shallow to Deep Learning

The **'hyper-sinh'** is a Python custom activation function available for both shallow and deep neural networks in TensorFlow and Keras for Machine Learning- and Deep Learning-based classification. It is distributed under the [CC BY 4.0 license](http://creativecommons.org/licenses/by/4.0/).

Details on this function, implementation and validation against gold standard activation functions for both shallow and deep neural networks are available at the following paper: **[Parisi et al., 2021a](https://www.sciencedirect.com/science/article/pii/S2666827021000566)**. 


### Dependencies

The dependencies are included in the `environment.yml` file. 
Run the following command to install the required version of Python (v3.9.16) and all dependencies in a conda virtual 
environment (replace `<env_name>` with your environment name):

- `conda env create --name <env_name> --file environment.yml`

### Usage

You can use the custom `HyperSinh` activation function in Keras as a layer:

#### Example of usage in a sequential model in Keras with a HyperSinh layer between a convolutional layer and a pooling layer

Either

```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(HyperSinh()) 
model.add(layers.MaxPooling2D((2, 2)))
```

or

```python
model = keras.Sequential(
        keras.Input(shape=(32, 32, 3)),

        layers.Conv2D(32, kernel_size=(3, 3)),
        HyperSinh(),

        layers.MaxPooling2D(pool_size=(2, 2)),
    ]
)
```

### Linting
`isort` is used to ensure a consistent order of imports, whilst `autopep8` to ensure adherence of the codes to PEP-8, 
via the following two commands respectively:

- `isort <folder_name>`
- `autopep8 --in-place --recursive .`

### Citation request

If you are using this function, please cite the papers by:
* **[Parisi et al., 2020](https://arxiv.org/abs/2011.07661)**.
* **[Parisi et al., 2021a](https://www.sciencedirect.com/science/article/pii/S2666827021000566)**.
* **[Parisi et al., 2021b](https://www.wseas.org/multimedia/journals/computerresearch/2021/a025118-001(2021).pdf)**.
