import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPooling2D

import cv2
import numpy as np

class Visualizer():
    def __init__(self, model) -> None:
        self.layers = model.layers

    def get_feature_maps(self, x):
        feature_maps = []
        for layer in self.layers:
            x = layer(x)
            if layer.__class__ == Conv2D:
                feature_maps.append(np.array(x))
        return feature_maps
    
    def transform_feature_maps(self, feature_maps):
        transformed_feature_maps = []
        for feature_map in feature_maps:
            transformed_feature_map = []
            for num in range(feature_map.shape[-1]):
                transformed_feature_map.append(np.array(feature_map[0,:,:,num]))
            transformed_feature_maps.append(np.array(transformed_feature_map))
        return transformed_feature_maps
    
    def apply_convolution(self, image, kernel=(3,3)):
        image_height = image.shape[0]
        image_width = image.shape[1]
        
        kernel_height, kernel_width = kernel

        half_kh = kernel_height//2
        half_kw = kernel_width//2

        padded_image = np.pad(image, (half_kh,half_kw), "constant", constant_values=(0,0))
        
        output_image = np.zeros(image.shape)

        for x in range(half_kh, image_height-(half_kh)):
            for y in range(half_kw, image_width-(half_kw)):
                sum = padded_image[
                    x-half_kh:x-half_kh+kernel_height,
                    y-half_kw:y-half_kw+kernel_width
                ]
                output_image[x][y] = sum.sum()
        return output_image

    def apply_gaussian_blur():
        pass
    
    def show_layers(self):
        for layer in self.layers:
            print(layer)
        


    