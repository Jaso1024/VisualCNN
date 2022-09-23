# VisualCNN

A visualizer for the internals of a Convolutional Neural Network.

Convolutional neural networks do not look at images the same way humans do, this project allows for the creation of a singular visual for hundreds of feature maps.

# Example

Using Keras' MNIST dataset with a tensorflow model whose summary looks like the following:
 
```Model: "sequential"
                   _________________________________________________________________
                       Layer (type)            Output Shape              Param #
                   =================================================================
                     conv2d (Conv2D)        (None, 28, 28, 32)            832

                     conv2d_1 (Conv2D)      (None, 28, 28, 64)           18496
                                
               max_pooling2d (MaxPooling2D) (None, 14, 14, 64)             0
 
                    conv2d_2 (Conv2D)       (None, 14, 14, 64)           36928

                    flatten (Flatten)         (None, 12544)                0

                     dense (Dense)            (None, 128)              1605760

                     dense_1 (Dense)           (None, 10)                 1290

                   =================================================================
                                        Total params: 1,663,306
                                      Trainable params: 1,663,306
                                        Non-trainable params: 0
                   _________________________________________________________________
 ```

<p align="center">
Layer 1: Convolution heatmap:
</p>

<p align="center">
 <a href="https://ibb.co/LJ5czHw"><img src="https://i.ibb.co/fnvyYLR/heatmap1.png" alt="heatmap1" border="0" width="400"></a>
</p>


<p align="center">
Layer 2: Convolution heatmap:
</p>

<p align="center">
<a href="https://ibb.co/V2C9D7H"><img src="https://i.ibb.co/3BFdWDf/heatmap2.png" alt="heatmap2" border="0" width="400"></a>
</p>

<p align="center">
Layer 3: 2D Max Pooling heatmap:
</p>

<p align="center">
<a href="https://ibb.co/nBpf68L"><img src="https://i.ibb.co/2SJPWYk/Heatmap3.png" alt="Heatmap3" border="0" width="400"></a>
</p>


Note:
 - If a layer's output shape is smaller than 14x14, it becomes difficult to produce for the visualizer to produce understandable representations of the feature maps
 - Does not work with models in which a layers output is bigger (in shape) than the original image
 
 - Visualizer may be incompatible with models in which image is not square
 
 - CNN must be made with tensorflow

## Dependencies
- numpy v1.23.2
- pandas v1.4.3
- tensorflow v2.9.1
- keras v2.9.0
- cv2 v4.6.0
- PIL v9.2.0
- Matplotlib v3.6.0
- Seaborn 0.12.0
- warn 0.1.0
- statistics 1.0.3.5
- scikit-image 0.19.3

## Usage
Create a tensorflow model

```
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Input, Flatten
from keras.callbacks import EarlyStopping
import cv2
import numpy as np
import matplotlib.pyplot as plt

num_classes = 10
input_shape = (28, 28, 1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (5,5), 1, padding="same"))
model.add(Conv2D(64, (3,3), 1, padding="same"))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3,3), 1, padding="same"))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10, batch_size=128, verbose=1)
```

Use the visualizer
```
from Vizualizer import Visualizer

visualizer = Visualizer(model, small_layers=True)
heatmaps = visualizer.heatmap(x_test[0])
visualizer.save_heatmaps()
```









