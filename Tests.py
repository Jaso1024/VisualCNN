import unittest
import pytest
from Vizualizer import Visualizer
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPooling2D

import cv2
import numpy as np

class SequentialVisualizerTests(unittest.TestCase):

    
    def setUp(self) -> None:
        model = Sequential()
        model.add(Conv2D(32,5,strides=2,activation="relu"))
        model.add(Conv2D(64,3,activation="relu"))
        model.add(Dense(8, activation="relu"))
        self.visualizer = Visualizer(model)

    def test_gen_feature_map_images(self):
        def test_gen_feature_maps():
            image = np.random.rand(1,64,64,3)
            feature_maps = self.visualizer.get_feature_maps(image)
            feature_map_shapes = [feature_map.shape for feature_map in feature_maps]
            for shape_num in range(len(feature_map_shapes)):
                self.assertEqual(len(feature_map_shapes[shape_num]), 4, f"Expected Convolution layer {shape_num+1} to return tensor with a rank of four, got tensor with shape {feature_map_shapes[shape_num]}")
            return feature_maps

        def test_transform_feature_maps(feature_maps):
            transformed_feature_maps = self.visualizer.transform_feature_maps(feature_maps)
            transformed_feature_map_shapes = [transformed_feature_map.shape for transformed_feature_map in transformed_feature_maps]
            for shape_num in range(len(transformed_feature_map_shapes)):
                self.assertEqual(len(transformed_feature_map_shapes[shape_num]), 3, f"Expected images to be an array of 2-dimensional arrays, got numpy array with shape {transformed_feature_map_shapes[shape_num]}")
        
        feature_maps = test_gen_feature_maps()
        test_transform_feature_maps(feature_maps)

    def test_apply_convolution(self):
        image = np.ones((32,32))
        image_after_conv = self.visualizer.apply_convolution(image)
        self.assertEqual(image_after_conv[1,1], 4.0)
        self.assertEqual(image_after_conv[30,30], 9.0)

unittest.main()