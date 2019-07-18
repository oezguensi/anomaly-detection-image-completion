import tensorflow as tf
import keras.backend as K

from keras.models import Model
from keras.layers import Input, Conv2D, Lambda, UpSampling2D, ELU


"""
Model parameters from the original paper: https://arxiv.org/pdf/1811.06861.pdf 
"""
input_shape = (128, 128, 1)

conv_layer_datas = [
    {
        'kernel_size': 5,
        'dilation_rate': 1,
        'strides': 1,
        'filters': 32
    },
    {
        'kernel_size': 3,
        'dilation_rate': 1,
        'strides': 1,
        'filters': 64
    },
    {
        'kernel_size': 3,
        'dilation_rate': 1,
        'strides': 1,
        'filters': 64
    },
    {
        'kernel_size': 3,
        'dilation_rate': 1,
        'strides': 2,
        'filters': 128
    },
    {
        'kernel_size': 3,
        'dilation_rate': 1,
        'strides': 1,
        'filters': 128
    },
    {
        'kernel_size': 3,
        'dilation_rate': 1,
        'strides': 1,
        'filters': 128
    },
    {
        'kernel_size': 3,
        'dilation_rate': 2,
        'strides': 1,
        'filters': 128
    },
    {
        'kernel_size': 3,
        'dilation_rate': 4,
        'strides': 1,
        'filters': 128
    },
    {
        'kernel_size': 3,
        'dilation_rate': 8,
        'strides': 1,
        'filters': 128
    },
    {
        'kernel_size': 3,
        'dilation_rate': 16,
        'strides': 1,
        'filters': 128
    },
    {
        'kernel_size': 3,
        'dilation_rate': 1,
        'strides': 1,
        'filters': 128
    },
    {
        'kernel_size': 3,
        'dilation_rate': 1,
        'strides': 1,
        'filters': 128
    },
    {
        'kernel_size': 3,
        'dilation_rate': 1,
        'strides': 1,
        'filters': 64
    },
    {
        'kernel_size': 3,
        'dilation_rate': 1,
        'strides': 1,
        'filters': 64
    },
    {
        'kernel_size': 3,
        'dilation_rate': 1,
        'strides': 1,
        'filters': 32
    },
    {
        'kernel_size': 3,
        'dilation_rate': 1,
        'strides': 1,
        'filters': 16
    },
    {
        'kernel_size': 3,
        'dilation_rate': 1,
        'strides': 1,
        'filters': 1
    }
]


def create_anomaly_cnn(input_shape=input_shape, conv_layer_datas=conv_layer_datas, model_width=1):
    """
    Creates the CNN used for Anomaly Detection.
    :param input_shape: The shape of the inputed image
    :param conv_layer_datas: The layer parameters of the model
    :return: Returns the model
    """

    def conv_block(outputs, kernel, dilation, strides, filters, model_width):
        outputs = Lambda(
            lambda x: tf.pad(x, [[0, 0], [dilation, dilation], [dilation, dilation], [0, 0]], 'REFLECT'))(outputs)
        outputs = Conv2D(filters=filters * model_width, kernel_size=kernel, strides=strides, padding='valid',
                         dilation_rate=dilation, activation='linear')(outputs)
        return ELU()(outputs)

    assert len(input_shape) == 2 or input_shape[-1] == 1, 'Images must only have one channel (grayscale)!'
    inputs = Input(shape=(*input_shape[:2], 1))
    outputs = inputs
    for i, data in enumerate(conv_layer_datas):
        outputs = conv_block(outputs, data['kernel_size'], data['dilation_rate'], data['strides'], data['filters'],
                             model_width if i != len(conv_layer_datas) - 1 else 1)
        outputs = UpSampling2D(size=2)(outputs) if i == 11 else outputs
    outputs = Lambda(lambda x: K.clip(x, -1, 1), name='clip')(outputs)
    outputs = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT'))(outputs)

    return Model(inputs=inputs, outputs=outputs)