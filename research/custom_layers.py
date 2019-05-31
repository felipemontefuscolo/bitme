# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Core Keras layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import nn
from numpy import prod
import numpy as np
from solver import solve

class RELUInitializer:
    def __init__(self, x, y, units=None):
        """
        Get initial state for RELU layer
        x.shape() = (# features, # training)
        """
        n=units
        if n is None:
            n = 2 * x.shape[0]
        box = np.array([x.min(axis=1), x.max(axis=1)]).transpose()
        assert box.shape == (x.shape[0], 2)
        k = 0
        A = []
        b = []
        n_boxes = n // (2 * x.shape[0])
        delta = .5 * (box[:, 1] - box[:, 0]) / n_boxes;
        while True:
            for side in range(x.shape[0]):
                for direction in (1,-1):
                    if k >= n:
                        break
                    k += 1
                    A.append(np.zeros(x.shape[0]))
                    A[-1][side] = float(direction)
                    idx = -(direction - 1) // 2
                    b.append(-direction * box[side][idx])
            if k >= n:
                break
            assert(all(delta >= 0.))
            box[:,0] += delta
            box[:,1] -= delta
            assert(all(box[:, 1] - box[:, 0] >= -1.e-14))
            
        self.A = np.array(A).transpose()
        self.b = np.array(b)
    
        a = self.A.transpose().dot(x) + self.b.reshape(n,1)
        C, d = solve(a, y)
        
        self.C = C.transpose()
        self.d = np.array(list(d))
        
        pass
    
    def get_A(self, shape, dtype=None, partition_info=None):
        return self._get(self.A, shape, dtype, partition_info)
    
    def get_b(self, shape, dtype=None, partition_info=None):
        return self._get(self.b, shape, dtype, partition_info)
    
    def get_C(self, shape, dtype=None, partition_info=None):
        return self._get(self.C, shape, dtype, partition_info)
    
    def get_d(self, shape, dtype=None, partition_info=None):
        return self._get(self.d, shape, dtype, partition_info)
    
    def _get(self, self_data, shape, dtype=None, partition_info=None):
        shape = tuple(shape)
        if self_data.shape != shape:
            raise ValueError('Shapes differ: (this) {} != {} (arg)'.format(self_data.shape, shape))
        return ops.convert_to_tensor(self_data, dtype=dtype)
    
class BiasOnly(Layer):
    def __init__(self,
                 units,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 activity_regularizer=None,
                 bias_constraint=None):
        units = int(units)
        input_shape = (units,)

        super(BiasOnly, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), input_shape=input_shape)
        self.units = int(units)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking = True
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `BiasOnly` '
                             'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2,
                                    axes={-1: last_dim})
        self.bias = self.add_weight(
            'bias',
            shape=[self.units, ],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=self.dtype,
            trainable=True)
        self.built = True

    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs)
        outputs = nn.bias_add(inputs, self.bias)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = {
            'units': self.units,
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(BiasOnly, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
