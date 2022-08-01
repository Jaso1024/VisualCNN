import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Input, Flatten
from keras.callbacks import EarlyStopping
import cv2
import numpy as np
import skimage
import statistics as stats
from Vizualizer import Visualizer
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

model = load_model("MnistCNN")

visualizer = Visualizer(model)
for num in range(10):
    visualizer.heatmap(x_test[num])
    visualizer.show_heatmap()

