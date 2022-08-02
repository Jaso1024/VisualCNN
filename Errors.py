import warnings

class SmallConvolutionWarning(UserWarning):
    pass

class OutputBiggerThanInputError(Exception):
    pass

class IncompatibleModelError(Exception):
    pass

class ImageTooSmallError(Exception):
    pass

class Image1DError(Exception):
    pass

class NanLayerOutputError(Exception):
    pass

class InvalidTypeError(Exception):
    pass

class SizeTooSmall(Exception):
    pass

