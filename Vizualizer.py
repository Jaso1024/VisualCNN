import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
import cv2
import numpy as np
import skimage
import statistics as stats
import seaborn as sb
import matplotlib.pyplot as plt
import math
import warnings
from Errors import IncompatibleModelError, ImageTooSmallError, Image1DError, NanLayerOutputError, InvalidTypeError, SizeTooSmall, SmallConvolutionWarning

class Visualizer():
    def __init__(self, model, small_layers=False) -> None:
        self.layers = model.layers
        self.upscale_even = UpSampling2D((2,2), interpolation="bilinear")
        self.upscale_odd = UpSampling2D((3,3), interpolation="bilinear")
        self.blur_map = []
        self.heat_map = []
        self.small_layers = small_layers

    def get_feature_maps(self, x):
        feature_maps = []
        for num in range(len(self.layers)):
            layer = self.layers[num]
            x = layer(x)
            if layer.__class__ == Conv2D or layer.__class__ == MaxPooling2D:
                if np.isnan(np.array(x).all()):
                    raise NanLayerOutputError
                elif (layer.output_shape[1] < 14 or layer.output_shape[2] < 14) and not self.small_layers:
                    warnings.warn(f"Layer {num} has an output shape too small to derive meaningful insight from - in most cases - "
                                   "and as such, will not be included in the final visual, if you would like layers with small output shapes to be included regardless, "
                                   "use the 'small_layers = True' argument when defining the Visualizer object", SmallConvolutionWarning, stacklevel=2)
                    continue
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
   
    def get_sigma(self, num):
        return 0.3*num

    def get_importance_images(self, image):
        image = np.squeeze(image)
        sigma_width_scalar = image.shape[1] / image.shape[0]
        for num in range(11,0,-1):
            sigma = self.get_sigma(num)
            yield skimage.filters.gaussian(image, sigma=(sigma, sigma*sigma_width_scalar))
        yield image 

    def get_mean_diffs(self, image):
        mean = stats.mean(image.flatten())
        mean_diffs = np.abs(image-mean)
        return mean_diffs
    
    def get_average_pixel_diffs(self, layer_diffs):
        summed_pixel_diffs = np.zeros(layer_diffs[0].shape)
        for diffs in layer_diffs:
            for x in range(len(diffs)):
                for y in range(len(diffs[0])):
                    diff = diffs[x][y]
                    summed_pixel_diffs[x][y] += diff
        return summed_pixel_diffs/len(layer_diffs)

    def get_pixel_importances(self, average_pixel_diffs):
        max = np.max(average_pixel_diffs)
        min = np.min(average_pixel_diffs)
        normed_diffs = (average_pixel_diffs-min)/(max-min)
        importances = np.round(normed_diffs, 4)
        importances = np.expand_dims(importances, axis=0)
        importances = np.expand_dims(importances, axis=-1)
        return importances

    def get_shape_diff(self, matrix1, matrix2):
        matrix1 = np.array(matrix1)
        matrix2 = np.array(matrix2)
        shape1 = np.array(matrix1.shape)
        shape2 = np.array(matrix2.shape)
        shape1_mid = int(len(shape1)/2-1)
        shape2_mid = int(len(shape2)/2-1)
        height_diff = (shape1[shape1_mid] - shape2[shape2_mid])**3
        width_diff = (shape1[shape1_mid+1] - shape2[shape2_mid+1])**3
        return (height_diff+width_diff)

    def upscale_importances(self, untampered_image, importances):
        current_shape_diff = self.get_shape_diff(untampered_image, importances)
        prev_shape_diff = current_shape_diff+1
        final_scaled = importances
        prev_scaled = final_scaled
        while prev_shape_diff >= current_shape_diff and current_shape_diff>=0:
            even_upscaled = np.array(self.upscale_even(final_scaled))
            odd_upscaled = np.array(self.upscale_odd(final_scaled))

            even_shape_diff = self.get_shape_diff(untampered_image, even_upscaled)
            odd_shape_diff = self.get_shape_diff(untampered_image, odd_upscaled)
            prev_shape_diff = current_shape_diff
            prev_scaled = final_scaled

            if any(diff>=0 for diff in (even_shape_diff, odd_shape_diff)):
                if even_shape_diff >= 0 and odd_shape_diff < 0:
                    current_shape_diff = even_shape_diff
                    final_scaled = even_upscaled
                elif odd_shape_diff >= 0 and even_shape_diff < 0:
                    current_shape_diff = odd_shape_diff
                    final_scaled = odd_upscaled
                else:
                    if even_shape_diff <= odd_shape_diff and even_shape_diff >= 0:
                        current_shape_diff = even_shape_diff
                        final_scaled = even_upscaled
                    elif odd_shape_diff < even_shape_diff and odd_shape_diff >= 0:
                        current_shape_diff = odd_shape_diff
                        final_scaled = odd_upscaled
            else:
                return prev_scaled
        return prev_scaled

    def transpose_importances(self, input_image, importances):
        while self.get_shape_diff(input_image, importances) != 0:
            importances = Conv2DTranspose(1,2)(importances)
        return importances

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)

    def norm_importances(self, importances):
        max = np.max(importances)
        min = np.min(importances)
        importances = (importances-min)/(max-min)
        return importances

    def map_importance_images(self, importances, importance_images):
        output_image = np.ones(importance_images[0].shape)
        for x in range(len(importance_images[0])):
            for y in range(len(importance_images[0][x])):
                importance = int(importances[0,x,y,0]*10)
                output_image[x][y] = importance_images[importance][x][y]
        return output_image

    def show_layers(self):
        for layer in self.layers:
            print(layer)
    
    def get_heatmap_gradient(self):
        gradient = []
        for num in range(1001):
            r = 0.3355*num + 2.236
            g = 1.6*num-1105
            b = int(2.7522*(math.pow(10,-9))*(math.pow(num,3.6499))-30.9771)

            rgb = [self.clamp(number, 0, 255) for number in (r,g,b)]
            gradient.append(np.array(rgb))
        return np.array(gradient)

    def handle_input(self, input):
        shape = input.shape
        image = np.array(input)
        if shape[len(shape)//2-1] <= 1 or shape[len(shape)//2] <= 1:
            raise ImageTooSmallError

        if len(shape) < 2:
            raise Image1DError
        elif len(shape) == 2:
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=-1)    
        elif len(shape) == 3:
            if shape[0] == 1:
                image = np.expand_dims(image, axis=-1)
            else:
                image = np.expand_dims(image, axis=0)
        return image
    
    def map_importance_colors(self, importances, gradient):
        output_image = np.ones([*importances.shape[1:3],3])
        for x in range(importances.shape[1]):
            for y in range(importances.shape[2]):
                importance = int(importances[0,x,y,0]*1000)
                output_image[x][y] = gradient[importance]
        return output_image

    def map(self, input, type):
        if type == self.heatmap:
            self.heat_map = []
        elif type == self.blurmap:
            self.blur_map = []
        else:
            raise InvalidTypeError

        input = self.handle_input(input)
        feature_maps = self.get_feature_maps(input)
        importance_images = list(self.get_importance_images(input))
        # list of numpy arrays, each array corresponding to a convolution layers output 
        post_conv_feature_maps = self.transform_feature_maps(feature_maps)
        output_images = []
        for layer_num in range(len(post_conv_feature_maps)):
            layer = post_conv_feature_maps[layer_num]
            layer_diffs = []
            for image in layer:
                mean_diffs = self.get_mean_diffs(image)
                layer_diffs.append(mean_diffs)
            average_pixel_diffs = self.get_average_pixel_diffs(layer_diffs)
            pixel_importances = self.get_pixel_importances(average_pixel_diffs)
            upscaled_importances = self.upscale_importances(input, pixel_importances)
            upscaled_importances = self.transpose_importances(input, upscaled_importances)
            normed_importances = self.norm_importances(upscaled_importances)
            if type == self.heatmap:
                output_images.append(np.squeeze(normed_importances))
                self.heat_map.append(np.squeeze(normed_importances))
            elif type == self.blurmap:
                importance_blurred_image = self.map_importance_images(normed_importances, importance_images)
                output_images.append(importance_blurred_image)
                self.blur_map.append(importance_blurred_image)
        return output_images

    def heatmap(self, input):
        return self.map(input, self.heatmap)

    def blurmap(self, input):
        return self.map(input, self.blurmap)
    
    def save_heatmaps(self):
        for idx in range(len(self.heat_map)):
            sb.heatmap(self.heat_map[idx])
            plt.savefig(f'Heatmap{idx+1}')
            plt.clf()

    def show_heatmap(self):
        for item in self.heat_map:
            sb.heatmap(item)
            plt.show()

    def show_blurmap(self, resize=500):
        for num in range(len(self.blur_map)):
            item = self.blur_map[num]
            shape = item.shape
            scalar = shape[1]/shape[0]

            if resize is not None and resize >=100:
                item = cv2.resize(item, (int(resize), int(resize*scalar)))
            elif resize is not None:
                raise SizeTooSmall
            cv2.imshow(f"Layer{num} Blurmap", item)
            cv2.waitKey(0)



