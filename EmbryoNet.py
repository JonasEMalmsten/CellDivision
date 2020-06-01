# -*- coding: utf-8 -*-
"""
This version of EmbryoNet is s copy of Keras' InceptionV3 model.
It has been refactored into a class, and is kept as a separate file so I can make changes
without editing the Keras code. So far, all changes I have tested has not improved the model,
so it's just a "copy" for now.
"""

# Much of this code is copied and modified from Keras. Their license is copied below, and shall apply to this work as well.
#
# COPYRIGHT
#
# All contributions by François Chollet:
# Copyright (c) 2015 - 2019, François Chollet.
# All rights reserved.
#
# All contributions by Google:
# Copyright (c) 2015 - 2019, Google, Inc.
# All rights reserved.
#
# All contributions by Microsoft:
# Copyright (c) 2017 - 2019, Microsoft, Inc.
# All rights reserved.
#
# All other contributions:
# Copyright (c) 2015 - 2019, the respective contributors.
# All rights reserved.
#
# Each contributor holds copyright over their respective contributions.
# The project versioning (Git) records all such contribution source information.
#
# LICENSE
#
# The MIT License (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import print_function
from __future__ import absolute_import

import os
import warnings

from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense, Reshape, Add, Multiply
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications import imagenet_utils
from keras.applications.imagenet_utils import decode_predictions
#from keras.applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import _obtain_input_shape
from batch_renorm import BatchRenormalization
from group_norm import GroupNormalization
from math import gcd

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

class EmbryoNet(Model):

    def conv2d_bn(self, x,
                  filters,
                  num_row,
                  num_col,
                  padding='same',
                  strides=(1, 1),
                  name=None):
        """Utility function to apply conv + BN.

        # Arguments
            x: input tensor.
            filters: filters in `Conv2D`.
            num_row: height of the convolution kernel.
            num_col: width of the convolution kernel.
            padding: padding mode in `Conv2D`.
            strides: strides in `Conv2D`.
            name: name of the ops; will become `name + '_conv'`
                for the convolution and `name + '_bn'` for the
                batch norm layer.

        # Returns
            Output tensor after applying `Conv2D` and `BatchNormalization`.
        """
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 3
        x = Conv2D(
            filters, (num_row, num_col),
            strides=strides,
            padding=padding,
            use_bias=False,
            name=conv_name)(x)
        if self.normalization == 'batch':
            x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        elif self.normalization == 'renorm':
#            x = BatchRenormalization(axis=bn_axis, scale=False, name=bn_name)(x)
            x = BatchRenormalization(axis=-1, scale=False, name=bn_name)(x)
        elif self.normalization == 'group':
#            x = GroupNormalization(axis=bn_axis, scale=False, name=bn_name, groups=8)(x)
            # Try to use 32 groups, but must be a an even divisor of filters
            groups=filters
            while groups>32 and groups%2==0:
                groups //= 2

            x = GroupNormalization(axis=bn_axis, scale=False, name=bn_name, groups=groups)(x)
        x = Activation('relu', name=name)(x)
        return x

    def EmbryoNetLayer(self, img_input, pooling=None, channel_axis=3, prefix_name=''):
        x = self.conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
        x = self.conv2d_bn(x, 32, 3, 3, padding='valid')
        x = self.conv2d_bn(x, 64, 3, 3)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv2d_bn(x, 80, 1, 1, padding='valid')
        x = self.conv2d_bn(x, 192, 3, 3, padding='valid')
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # mixed 0, 1, 2: 35 x 35 x 256
        branch1x1 = self.conv2d_bn(x, 64, 1, 1)

        branch5x5 = self.conv2d_bn(x, 48, 1, 1)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 32, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name=prefix_name+'mixed0')

        self.layers_mixed.append(x)

        # mixed 1: 35 x 35 x 256
        branch1x1 = self.conv2d_bn(x, 64, 1, 1)

        branch5x5 = self.conv2d_bn(x, 48, 1, 1)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name=prefix_name+'mixed1')

        self.layers_mixed.append(x)

        # mixed 2: 35 x 35 x 256
        branch1x1 = self.conv2d_bn(x, 64, 1, 1)

        branch5x5 = self.conv2d_bn(x, 48, 1, 1)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name=prefix_name+'mixed2')

        self.layers_mixed.append(x)

        # mixed 3: 17 x 17 x 768
        branch3x3 = self.conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self.conv2d_bn(
            branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate(
            [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name=prefix_name+'mixed3')

        self.layers_mixed.append(x)


        # mixed 4: 17 x 17 x 768
        branch1x1 = self.conv2d_bn(x, 192, 1, 1)

        branch7x7 = self.conv2d_bn(x, 128, 1, 1)
        branch7x7 = self.conv2d_bn(branch7x7, 128, 1, 7)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = self.conv2d_bn(x, 128, 1, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 1, 7)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name=prefix_name+'mixed4')

        self.layers_mixed.append(x)

# test
#        self.hpfin = Input(shape=(1,))
#        hpf = Dense(12*12*768, activation='relu')(self.hpfin)
#        hpf = Reshape((12,12,768))(hpf)
#        x = Add()([x, hpf])

        # mixed 5, 6: 17 x 17 x 768
        for i in range(2):
            branch1x1 = self.conv2d_bn(x, 192, 1, 1)

            branch7x7 = self.conv2d_bn(x, 160, 1, 1)
            branch7x7 = self.conv2d_bn(branch7x7, 160, 1, 7)
            branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = self.conv2d_bn(x, 160, 1, 1)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 1, 7)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.concatenate(
                [branch1x1, branch7x7, branch7x7dbl, branch_pool],
                axis=channel_axis,
                name=prefix_name+'mixed' + str(5 + i))

            self.layers_mixed.append(x)

        # mixed 7: 17 x 17 x 768
        branch1x1 = self.conv2d_bn(x, 192, 1, 1)

        branch7x7 = self.conv2d_bn(x, 192, 1, 1)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 1, 7)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = self.conv2d_bn(x, 192, 1, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name=prefix_name+'mixed7')

        self.layers_mixed.append(x)

        # mixed 8: 8 x 8 x 1280
        branch3x3 = self.conv2d_bn(x, 192, 1, 1)
        branch3x3 = self.conv2d_bn(branch3x3, 320, 3, 3,
                              strides=(2, 2), padding='valid')

        branch7x7x3 = self.conv2d_bn(x, 192, 1, 1)
        branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, 1, 7)
        branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, 7, 1)
        branch7x7x3 = self.conv2d_bn(
            branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate(
            [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name=prefix_name+'mixed8')

        self.layers_mixed.append(x)

        # mixed 9: 8 x 8 x 2048
        for i in range(2):
            branch1x1 = self.conv2d_bn(x, 320, 1, 1)

            branch3x3 = self.conv2d_bn(x, 384, 1, 1)
            branch3x3_1 = self.conv2d_bn(branch3x3, 384, 1, 3)
            branch3x3_2 = self.conv2d_bn(branch3x3, 384, 3, 1)
            branch3x3 = layers.concatenate(
                [branch3x3_1, branch3x3_2], axis=channel_axis, name=prefix_name+'mixed9_' + str(i))

            branch3x3dbl = self.conv2d_bn(x, 448, 1, 1)
            branch3x3dbl = self.conv2d_bn(branch3x3dbl, 384, 3, 3)
            branch3x3dbl_1 = self.conv2d_bn(branch3x3dbl, 384, 1, 3)
            branch3x3dbl_2 = self.conv2d_bn(branch3x3dbl, 384, 3, 1)
            branch3x3dbl = layers.concatenate(
                [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

            branch_pool = AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.concatenate(
                [branch1x1, branch3x3, branch3x3dbl, branch_pool],
                axis=channel_axis,
                name=prefix_name+'mixed' + str(9 + i))

            self.layers_mixed.append(x)


        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

        return x

    def __init__(self, include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                normalization='batch'):
        """Instantiates the Embryo Inception v3 architecture.
        nb_axis_modifier increase the axis for batch normalization. For example, set to 1 if layer is wrapped in TimeDistributed
        # Returns
            A Keras model instance.

        # Raises
            ValueError: in case of invalid argument for `weights`,
                or invalid input shape.
        """
        self.normalization = normalization
        self.layers_mixed = [] # easy access to mixed layers

        if not (weights in {'imagenet', None} or os.path.exists(weights)):
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization), `imagenet` '
                             '(pre-training on ImageNet), '
                             'or the path to the weights file to be loaded.')

        # Determine proper input shape
        input_shape = _obtain_input_shape(
            input_shape,
            default_size=299,
            min_size=139,
            data_format=K.image_data_format(),
            require_flatten=False,
            weights=weights)

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3

        x = self.EmbryoNetLayer(img_input, pooling=pooling, channel_axis=channel_axis)

        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input

#        super(EmbryoNet, self).__init__([inputs, self.hpfin], x, name='EmbryoNet')
        super(EmbryoNet, self).__init__(inputs, x, name='EmbryoNet')
#        model = Model(inputs, x, name='inception_v3')
#        return model


def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, mode='tf')


