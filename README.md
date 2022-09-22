# VisualCNN

A project created for the visualizaton of the feature extraction of the inner nodes of a Concolutional Neural Network (CNN).

# Example











```Model: "sequential"
________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 32)        320

 conv2d_1 (Conv2D)           (None, 28, 28, 64)        18496

 conv2d_2 (Conv2D)           (None, 28, 28, 64)        36928

 conv2d_3 (Conv2D)           (None, 28, 28, 128)       73856

 flatten (Flatten)           (None, 100352)            0

 dense (Dense)               (None, 128)               12845184

 dense_1 (Dense)             (None, 10)                1290

=================================================================
Total params: 12,976,074
Trainable params: 12,976,074
Non-trainable params: 0
_________________________________________________________________
```






Layer:

Note:
 - Does not work with models in which a layers output is bigger (in shape) than the original image
 - May be incompatible with models in which image is not square
