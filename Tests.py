from turtle import up
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
        self.model = Sequential()
        self.model.add(Conv2D(32,2, activation="relu"))
        self.model.add(Conv2D(64,2,strides=(2,2), activation="relu"))
        self.model.add(Conv2D(64,2, activation="relu"))
        self.model.add(Dense(8,activation="relu"))
        self.visualizer = Visualizer(self.model)

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
    
    def test_get_mean_diffs(self):
        image = np.ones((320,320))
        mean_diffs = self.visualizer.get_mean_diffs(image)
        self.assertEqual(mean_diffs.flatten()[0], 0)
    
    def test_get_average_pixel_diffs(self):
        layer_diffs = []
        for num in range(8):
            arr = np.random.rand(320, 320)
            arr.fill(0.5)
            layer_diffs.append(arr)
        average_pixel_diffs = self.visualizer.get_average_pixel_diffs(layer_diffs)
        self.assertEqual(average_pixel_diffs.flatten()[0], 0.5)
    
    def test_get_pixel_importances(self):
        diffs = np.random.rand(320,320)
        importances = self.visualizer.get_pixel_importances(diffs)
        
    def test_get_importance_images(self):
        image = np.random.rand(320,320)
        importance_images = self.visualizer.get_importance_images(image)

    def test_upscale_importances(self):
        image = np.random.rand(320,320)
        diffs = np.random.rand(16,16)
        importances = self.visualizer.get_pixel_importances(diffs)
        upscaled = self.visualizer.upscale_importances(image, importances)
        self.assertEqual(upscaled.shape, (1, 288, 288, 1))
    
    def test_transpose_importances(self):
        image = np.random.rand(1, 316, 316, 1)
        input_image = np.random.rand(1, 320, 320, 1)
        transposed_importances = self.visualizer.transpose_importances(input_image, image, self.model.layers)
        self.assertEqual(input_image.shape, transposed_importances.shape)

    def test_map_importance_images(self):
        importances = self.visualizer.get_pixel_importances(np.random.rand(320, 320))
        image = np.random.rand(320, 320)
        importance_images = [image for image in self.visualizer.get_importance_images(image)]
        output = self.visualizer.map_importance_images(importances, importance_images)
    
unittest.main()