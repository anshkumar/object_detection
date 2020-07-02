# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""SSDFeatureExtractor for MobilenetV2 features."""

import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import context_manager
from object_detection.utils import ops
from object_detection.utils import shape_utils
from nets.mobilenet import mobilenet
from nets.mobilenet import mobilenet_v2

slim = contrib_slim

def _batch_norm_arg_scope(list_ops,
                          use_batch_norm=True,
                          batch_norm_decay=0.9997,
                          batch_norm_epsilon=0.001,
                          batch_norm_scale=False,
                          train_batch_norm=False):
  """Slim arg scope for InceptionV2 batch norm."""
  if use_batch_norm:
    batch_norm_params = {
        'is_training': train_batch_norm,
        'scale': batch_norm_scale,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon
    }
    normalizer_fn = slim.batch_norm
  else:
    normalizer_fn = None
    batch_norm_params = None

  return slim.arg_scope(list_ops,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=batch_norm_params)

class SSDMobileNetV2FeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """SSD Feature Extractor using MobilenetV2 features."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               num_layers=6,
               override_base_feature_extractor_hyperparams=False,
               batch_norm_trainable=False):
    """MobileNetV2 Feature Extractor for SSD Models.

    Mobilenet v2 (experimental), designed by sandler@. More details can be found
    in //knowledge/cerebra/brain/compression/mobilenet/mobilenet_experimental.py

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False.
      use_depthwise: Whether to use depthwise convolutions. Default is False.
      num_layers: Number of SSD layers.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    self._depth_multiplier = depth_multiplier
    self._min_depth = min_depth
    self._train_batch_norm = (batch_norm_trainable and is_training)
    super(SSDMobileNetV2FeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams_fn=conv_hyperparams_fn,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        num_layers=num_layers,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)

  def preprocess(self, resized_inputs):
    """SSD preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (2.0 / 255.0) * resized_inputs - 1.0

  def extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    preprocessed_inputs = shape_utils.check_min_image_dim(
        33, preprocessed_inputs)

    feature_map_layout = {
        'from_layer': ['layer_15/expansion_output', 'layer_19', '', '', '', ''
                      ][:self._num_layers],
        'layer_depth': [-1, -1, 512, 256, 256, 128][:self._num_layers],
        'use_depthwise': self._use_depthwise,
        'use_explicit_padding': self._use_explicit_padding,
    }

    with tf.variable_scope('MobilenetV2', reuse=self._reuse_weights) as scope:
      with slim.arg_scope(
          mobilenet_v2.training_scope(is_training=None, bn_decay=0.9997)), \
          slim.arg_scope(
              [mobilenet.depth_multiplier], min_depth=self._min_depth):
        with (slim.arg_scope(self._conv_hyperparams_fn())
              if self._override_base_feature_extractor_hyperparams else
              context_manager.IdentityContextManager()):
          _, image_features = mobilenet_v2.mobilenet_base(
              ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple),
              final_endpoint='layer_19',
              depth_multiplier=self._depth_multiplier,
              use_explicit_padding=self._use_explicit_padding,
              scope=scope)
        with slim.arg_scope(self._conv_hyperparams_fn()):
          feature_maps = feature_map_generators.multi_resolution_feature_maps(
              feature_map_layout=feature_map_layout,
              depth_multiplier=self._depth_multiplier,
              min_depth=self._min_depth,
              insert_1x1_conv=True,
              image_features=image_features)

    return feature_maps.values()

  def _extract_box_classifier_features(self, proposal_feature_maps, scope):
    """Extracts second stage box classifier features.

    Args:
      proposal_feature_maps: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
        representing the feature map cropped to each proposal.
      scope: A scope name (unused).

    Returns:
      proposal_classifier_features: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, height, width, depth]
        representing box classifier features for each proposal.
    """
    net = proposal_feature_maps

    conv_depth = 1024
    if self._skip_last_stride:
      conv_depth_ratio = float(self._conv_depth_ratio_in_percentage) / 100.0
      conv_depth = int(float(conv_depth) * conv_depth_ratio)

    depth = lambda d: max(int(d * 1.0), 16)
    with tf.variable_scope('MobilenetV2', reuse=self._reuse_weights):
      with slim.arg_scope(
          mobilenet_v1.mobilenet_v1_arg_scope(
              is_training=self._train_batch_norm,
              weight_decay=self._weight_decay)):
        with slim.arg_scope(
            [slim.conv2d, slim.separable_conv2d], padding='SAME'):
          net = slim.separable_conv2d(
              net,
              depth(conv_depth), [3, 3],
              depth_multiplier=1,
              stride=2,
              scope='Conv2d_12_pointwise')
          return slim.separable_conv2d(
              net,
              depth(conv_depth), [3, 3],
              depth_multiplier=1,
              stride=1,
              scope='Conv2d_13_pointwise')

