import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, InputLayer
from define_models import augLayers, convXpress

def get_model(rnd_seed):
    """Builds the ML Model.

    Args:
        rnd_seed: Seed for the random functions.

    Returns:
        The ML Model.
    """
    # Make new sequential model
    model = Sequential()
    # Add Input Layer
    model.add(InputLayer(input_shape=(64,64,1)))
    # Add ConvXpress
    model.add(convXpress(rnd_seed,(64,64,1),4))

    return model