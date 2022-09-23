# VisualCNN

A project created to create visual representations of what a CNN sees, combining all of the feature maps a convolution layer outputs into 1 visualizatio.

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

## Quickstart

```
from Vizualizer import Visualizer

visualizer = Visualizer(model, small_layers=True)
heatmaps = visualizer.heatmap(x_test[0])
visualizer.save_heatmaps()
```









