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
"""Train a ResNet-50 model on ImageNet on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('/path/to/root')

import os
import time
import glob
import csv

from absl import app
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import
import numpy as np
#import tensorflow as tf
import re

import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
#import tensorflow_transform as tft

tf.disable_v2_behavior()

#from common import tpu_profiler_hook
#from official.resnet import imagenet_input2
import imagenet_input
#from official.resnet import lars_util
import lars_util
#from official.resnet import resnet_model
import resnet_model

#from tensorflow.contrib import summary
#from tensorflow.contrib.tpu.python.tpu import async_checkpoint
#from tensorflow.contrib.training.python.training import evaluation
#from tensorflow.core.protobuf import rewriter_config_pb2
#from tensorflow.python.estimator import estimator

from tensorflow.compat.v2 import summary
from tensorflow.core.protobuf import rewriter_config_pb2  # pylint: disable=g-direct-tensorflow-import
#from tensorflow.compat.v1.python.estimator import estimator


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = flags.FLAGS

PRICE_COUNT = 15 #12
DIMENSION_COUNT = 15 #10
CHANNEL_COUNT = 3
#LABEL_COUNT = 2 #Delete Turtle X 20210214 BorisTown
LABEL_COUNT = 10 #Insert Turtle X 20210214 BorisTown
#PREDICT_BATCH_SIZE = 31
#MAX_CASE = 10
GROUP_COUNT = 4
#price_list_len = 519
FAKE_DATA_DIR = 'gs://cloud-tpu-test-datasets/fake_imagenet'

flags.DEFINE_bool(
    'use_tpu', default=True,
    help=('Use TPU to execute the model for training and evaluation. If'
          ' --use_tpu=false, will use whatever devices are available to'
          ' TensorFlow by default (e.g. CPU and GPU)'))

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

# Model specific flags
flags.DEFINE_string(
    'data_dir', default=FAKE_DATA_DIR,
    help=('The directory where the ImageNet input data is stored. Please see'
          ' the README.md for the expected data format.'))

flags.DEFINE_string(
    'prices_dir', default=None,
    help=('The directory where the source prices for prediction is stored. Please see'
          ' the README.md for the expected data format.'))

flags.DEFINE_string(
    'predict_dir', default=None,
    help=('The directory where the prediction result is stored. Please see'
          ' the README.md for the expected data format.'))

flags.DEFINE_string(
    'model_dir', default=None,
    help=('The directory where the model and training/evaluation summaries are'
          ' stored.'))

flags.DEFINE_integer(
    'resnet_depth', default=50,
    help=('Depth of ResNet model to use. Must be one of {18, 34, 50, 101, 152,'
          ' 200, 1000}. ResNet-18 and 34 use the pre-activation residual blocks'
          ' without bottleneck layers. The other models use pre-activation'
          ' bottleneck layers. Deeper models require more training time and'
          ' more memory and may require reducing --train_batch_size to prevent'
          ' running out of memory.'))

flags.DEFINE_integer(
    'profile_every_n_steps', default=0,
    help=('Number of steps between collecting profiles if larger than 0'))

flags.DEFINE_string(
    'mode', default='train_and_eval',
    help='One of {"train_and_eval", "train", "eval", "predict", "save_model"}.')

flags.DEFINE_integer(
    'train_steps', default=112590,
    help=('The number of steps to use for training. Default is 112590 steps'
          ' which is approximately 90 epochs at batch size 1024. This flag'
          ' should be adjusted according to the --train_batch_size flag.'))

flags.DEFINE_integer(
    'train_batch_size', default=1024, help='Batch size for training.')

flags.DEFINE_integer(
    'eval_batch_size', default=1024, help='Batch size for evaluation.')

flags.DEFINE_integer(
    'input_batch_size', default=2000, help='Batch size for input.')

flags.DEFINE_integer(
    'num_train_images', default=1281167, help='Size of training data set.')

flags.DEFINE_integer(
    'num_eval_images', default=50000, help='Size of evaluation data set.')

flags.DEFINE_integer(
    'num_label_classes', default=10, help='Number of classes, at least 2')

flags.DEFINE_integer(
    'steps_per_eval', default=1251,
    help=('Controls how often evaluation is performed. Since evaluation is'
          ' fairly expensive, it is advised to evaluate as infrequently as'
          ' possible (i.e. up to --train_steps, which evaluates the model only'
          ' after finishing the entire training regime).'))

flags.DEFINE_integer(
    'eval_timeout',
    default=None,
    help='Maximum seconds between checkpoints before evaluation terminates.')

flags.DEFINE_bool(
    'skip_host_call', default=False,
    help=('Skip the host_call which is executed every training step. This is'
          ' generally used for generating training summaries (train loss,'
          ' learning rate, etc...). When --skip_host_call=false, there could'
          ' be a performance drop if host_call function is slow and cannot'
          ' keep up with the TPU-side computation.'))

flags.DEFINE_integer(
    'iterations_per_loop', default=1251,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

flags.DEFINE_integer(
    'num_parallel_calls', default=8,
    help=('Number of parallel threads in CPU for the input pipeline.'
          ' Recommended value is the number of cores per CPU host.'))

flags.DEFINE_integer(
    'num_cores', default=8,
    help=('Number of TPU cores in total. For a single TPU device, this is 8'
          ' because each TPU has 4 chips each with 2 cores.'))

flags.DEFINE_string(
    'bigtable_project', None,
    'The Cloud Bigtable project.  If None, --gcp_project will be used.')
flags.DEFINE_string(
    'bigtable_instance', None,
    'The Cloud Bigtable instance to load data from.')
flags.DEFINE_string(
    'bigtable_table', 'imagenet',
    'The Cloud Bigtable table to load data from.')
flags.DEFINE_string(
    'bigtable_train_prefix', 'train_',
    'The prefix identifying training rows.')
flags.DEFINE_string(
    'bigtable_eval_prefix', 'train_',
    'The prefix identifying evaluation rows.')
flags.DEFINE_string(
    'bigtable_column_family', 'tfexample',
    'The column family storing TFExamples.')
flags.DEFINE_string(
    'bigtable_column_qualifier', 'example',
    'The column name storing TFExamples.')

flags.DEFINE_string(
    'data_format', default='channels_last',
    help=('A flag to override the data format used in the model. The value'
          ' is either channels_first or channels_last. To run the network on'
          ' CPU or TPU, channels_last should be used. For GPU, channels_first'
          ' will improve performance.'))

# TODO(chrisying): remove this flag once --transpose_tpu_infeed flag is enabled
# by default for TPU
flags.DEFINE_bool(
    'transpose_input', default=True,
    help='Use TPU double transpose optimization')

flags.DEFINE_string(
    'export_dir',
    default=None,
    help=('The directory where the exported SavedModel will be stored.'))

flags.DEFINE_bool(
    'export_to_tpu', default=False,
    help=('Whether to export additional metagraph with "serve, tpu" tags'
          ' in addition to "serve" only metagraph.'))

flags.DEFINE_string(
    'precision', default='bfloat16',
    help=('Precision to use; one of: {bfloat16, float32}'))

flags.DEFINE_float(
    'base_learning_rate', default=0.1,
    help=('Base learning rate when train batch size is 256.'))

flags.DEFINE_float(
    'momentum', default=0.9,
    help=('Momentum parameter used in the MomentumOptimizer.'))

flags.DEFINE_float(
    'weight_decay', default=1e-4,
    help=('Weight decay coefficiant for l2 regularization.'))

flags.DEFINE_float(
    'label_smoothing', default=0.1,
    help=('Label smoothing parameter used in the softmax_cross_entropy'))

flags.DEFINE_integer('log_step_count_steps', 64, 'The number of steps at '
                     'which the global step information is logged.')

flags.DEFINE_bool('enable_lars',
                  default=False,
                  help=('Enable LARS optimizer for large batch training.'))

flags.DEFINE_float('poly_rate', default=0.0,
                   help=('Set LARS/Poly learning rate.'))

flags.DEFINE_bool(
    'use_cache', default=True, help=('Enable cache for training input.'))

flags.DEFINE_bool(
    'use_async_checkpointing', default=False, help=('Enable async checkpoint'))

flags.DEFINE_integer('image_size', 4, 'The input image size.')

flags.DEFINE_string(
    'dropblock_groups', '1,2',
    #'dropblock_groups', '',
    help=('A string containing comma separated integers indicating ResNet '
          'block groups to apply DropBlock. `3,4` means to apply DropBlock to '
          'block groups 3 and 4. Use an empty string to not apply DropBlock to '
          'any block group.'))
flags.DEFINE_float(
    'dropblock_keep_prob', default=0.5,
    help=('keep_prob parameter of DropBlock. Will not be used if '
          'dropblock_groups is empty.'))
flags.DEFINE_integer(
    'dropblock_size', default=3,
    help=('size parameter of DropBlock. Will not be used if dropblock_groups '
          'is empty.'))


# The input tensor is in the range of [0, 255], we need to scale them to the
# range of [0, 1]
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def get_lr_schedule(train_steps, num_train_images, train_batch_size):
  """learning rate schedule."""
  steps_per_epoch = np.floor(num_train_images / train_batch_size)
  train_epochs = train_steps / steps_per_epoch
  return [  # (multiplier, epoch to start) tuples
      (1.0, np.floor(5 / 90 * train_epochs)),
      (0.1, np.floor(30 / 90 * train_epochs)),
      (0.01, np.floor(60 / 90 * train_epochs)),
      (0.001, np.floor(80 / 90 * train_epochs))
  ]


def learning_rate_schedule(train_steps, current_epoch):
  """Handles linear scaling rule, gradual warmup, and LR decay.

  The learning rate starts at 0, then it increases linearly per step.
  After 5 epochs we reach the base learning rate (scaled to account
    for batch size).
  After 30, 60 and 80 epochs the learning rate is divided by 10.
  After 90 epochs training stops and the LR is set to 0. This ensures
    that we train for exactly 90 epochs for reproducibility.

  Args:
    train_steps: `int` number of training steps.
    current_epoch: `Tensor` for current epoch.

  Returns:
    A scaled `Tensor` for current learning rate.
  """
  scaled_lr = FLAGS.base_learning_rate * (FLAGS.train_batch_size / 256.0)

  lr_schedule = get_lr_schedule(
      train_steps=train_steps,
      num_train_images=FLAGS.num_train_images,
      train_batch_size=FLAGS.train_batch_size)
  decay_rate = (scaled_lr * lr_schedule[0][0] *
                current_epoch / lr_schedule[0][1])
  for mult, start_epoch in lr_schedule:
    decay_rate = tf.where(current_epoch < start_epoch,
                          decay_rate, scaled_lr * mult)
  return decay_rate

def feature_mirror(features):
  #shape:[Batch,Height,Width,Cannel]
  if FLAGS.precision == 'bfloat16':
    features_copy = tf.cast(features, tf.float32)
  elif FLAGS.precision == 'float32':
    features_copy = tf.identity(features)
  #features_copy = tf.subtract(1.0, features_copy)
  feature_copy = features_copy * -1.0 + 1.0
  if FLAGS.precision == 'bfloat16':
    features_copy = tf.cast(features_copy, tf.bfloat16)
  features_combine = tf.concat([features, features_copy], axis=0)
  #shape:[Batch*2,Height,Width,Cannel]
  return features_combine

def label_mirror(labels):
  #shape:[Max_Case,Batch,Label_Class]
  labels_copy = tf.identity(labels)
  #labels_copy = tf.subtract(1.0 ,labels_copy)
  labels_copy = labels_copy * -1 + 1
  labels_combine = tf.concat([labels, labels_copy], axis=1)
  #shape:[Max_Case,Batch*2,Label_Class]
  return labels_combine

def scale_to_0_1(x):
  # x is your tensor
  current_min = tf.reduce_min(x)
  current_max = tf.reduce_max(x)
  target_min = 0
  target_max = 1
  
  if current_max == current_min:
    return tf.ones(x.shape, dtype=tf.float32)
  # scale to [0; 1]
  x = (x - current_min) / (current_max - current_min)

  # scale to [target_min; target_max]
  x = x * (target_max - target_min) + target_min
  return x

def resnet_model_fn(features, labels, mode, params):
  """The model_fn for ResNet to be used with TPUEstimator.

  Args:
    features: `Tensor` of batched images. If transpose_input is enabled, it
        is transposed to device layout and reshaped to 1D tensor.
    labels: `Tensor` of labels for the data samples
    mode: one of `tf.estimator.ModeKeys.{TRAIN,EVAL,PREDICT}`
    params: `dict` of parameters passed to the model from the TPUEstimator,
        `params['batch_size']` is always provided and should be used as the
        effective batch size.

  Returns:
    A `TPUEstimatorSpec` for the model
  """
  if isinstance(features, dict):
    features = features['feature']
  
  '''
  #Comment by AI Era V2.1 Begin 20200315
  max_batch_len = FLAGS.input_batch_size #2000
  max_batch_len_tensor = tf.constant(max_batch_len, dtype=tf.int64)
  #Comment by AI Era V2.1 End 20200315
  '''

  # Insert Loop Code From Here Boris Town 20200109
  
  # In most cases, the default data format NCHW instead of NHWC should be
  # used for a significant performance boost on GPU/TPU. NHWC should be used
  # only if the network needs to be run on CPU since the pooling operations
  # are only supported on NHWC.
  if FLAGS.data_format == 'channels_first':
    assert not FLAGS.transpose_input    # channels_first only for GPU
    features = tf.transpose(features, [0, 3, 1, 2])

  #if FLAGS.transpose_input and mode != tf.estimator.ModeKeys.PREDICT:
  if FLAGS.transpose_input:
    features = tf.reshape(features, [PRICE_COUNT, DIMENSION_COUNT, CHANNEL_COUNT, -1])
    features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC
    #features = tf.reshape(features, [price_list_len, -1])
    #features = tf.transpose(features, [1, 0])  # [Price,Batch] to [Batch,Price]
    #labels = tf.reshape(labels, [price_list_len, -1])
    #labels = tf.transpose(labels, [1, 0])  # [Score,Batch] to [Batch,Score]
    if mode != tf.estimator.ModeKeys.PREDICT:
      labels = tf.reshape(labels, [FLAGS.num_label_classes, -1])
      labels = tf.transpose(labels, [1, 0])  # LN to NL
      tf.logging.info("features=%s,labels=%s" % (features.shape, labels.shape))
    
  #if FLAGS.transpose_input and mode == tf.estimator.ModeKeys.PREDICT:
  #  features = tf.reshape(features, [PRICE_COUNT, DIMENSION_COUNT, CHANNEL_COUNT, -1])
  #  features = tf.transpose(features, [3, 0, 1, 2])  # HWCN to NHWC

  # DropBlock keep_prob for the 8 block groups of ResNet architecture.
  # None means applying no DropBlock at the corresponding block group.
  dropblock_keep_probs = [None] * GROUP_COUNT
  if FLAGS.dropblock_groups:
    # Scheduled keep_prob for DropBlock.
    train_steps = tf.cast(FLAGS.train_steps, tf.float32)
    current_step = tf.cast(tf.train.get_global_step(), tf.float32)
    current_ratio = current_step / train_steps
    dropblock_keep_prob = (1 - current_ratio * (1 - FLAGS.dropblock_keep_prob))

    # Computes DropBlock keep_prob for different block groups of ResNet.
    dropblock_groups = [int(x) for x in FLAGS.dropblock_groups.split(',')]
    for block_group in dropblock_groups:
      if block_group < 1 or block_group > GROUP_COUNT:
        raise ValueError(
            'dropblock_groups should be a comma separated list of integers '
            'between 1 and GROUP_COUNT (dropblcok_groups: {}).'
            .format(FLAGS.dropblock_groups))
      dropblock_keep_probs[block_group - 1] = 1 - (
          (1.0 - dropblock_keep_prob) / GROUP_COUNT**(GROUP_COUNT - block_group))
  
  if mode != tf.estimator.ModeKeys.PREDICT:
    #features=feature_mirror(features)
    #labels=label_mirror(labels)
    tf.logging.info("features=%s,labels=%s" % (features.shape, labels.shape))
  
  # This nested function allows us to avoid duplicating the logic which
  # builds the network, for different values of --precision.
  def build_network(l_features):
    network = resnet_model.resnet_v1(
        resnet_depth=FLAGS.resnet_depth,
        num_classes=FLAGS.num_label_classes,
        dropblock_size=FLAGS.dropblock_size,
        dropblock_keep_probs=dropblock_keep_probs,
        data_format=FLAGS.data_format)
    return network(inputs=l_features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))
  
  #priceInputCount = PRICE_COUNT * DIMENSION_COUNT * CHANNEL_COUNT
  
  #if mode != tf.estimator.ModeKeys.PREDICT:
  #    if FLAGS.precision == 'bfloat16':
  #      labeltensor = tf.zeros([max_batch_len, 2], dtype=tf.bfloat16)
  #      pricestensor = tf.zeros([max_batch_len, priceInputCount], dtype=tf.bfloat16)
  #    else:
  #      labeltensor = tf.zeros([max_batch_len, 2], dtype=tf.float32)
  #      pricestensor = tf.zeros([max_batch_len, priceInputCount], dtype=tf.float32)

  #    batchCount = labels.shape[0]
  #    #labels_int = tf.cast(labels, tf.int64)
  #    labels_int = tf.count_nonzero(features, 1)
  #    arrayindex = tf.Variable(0, dtype=tf.int64, trainable=False)
  #    for batchIndex in range(batchCount):
  #      priceList = features[batchIndex]
  #      def make_training_set(arrayindex, labeltensor, pricestensor):
  #        #if labels[batchIndex] > tf.constant(priceInputCount):
  #        trainingCount = labels_int[batchIndex] - tf.constant(priceInputCount-1, dtype=tf.int64)
  #        trainingIndex = tf.identity(trainingCount) - 1
  #        def while_cond(arrayindex, trainingIndex, trainingCount, labeltensor, pricestensor):
  #          return tf.math.logical_and(trainingIndex >= 0, arrayindex < max_batch_len_tensor)
  #        def while_body(arrayindex, trainingIndex, trainingCount, labeltensor, pricestensor):
  #          #for trainingIndex in range(trainingCount):
  #          trainingInputData = tf.zeros([priceInputCount], dtype=tf.float32)
  #          for price_element_index in range(priceInputCount):
  #              one_hot_price_element = tf.one_hot(price_element_index, priceInputCount, on_value=1, off_value = 0, dtype=tf.int32, name="price_element")
  #              trainingInputData = tf.identity(trainingInputData + tf.cast(one_hot_price_element, tf.float32) * priceList[trainingIndex + priceInputCount - price_element_index - 1])
  #          trainingInputData = scale_to_0_1(trainingInputData)
  #          trainingInputData = tf.reshape(trainingInputData, [priceInputCount])
  #          trainingInputData_Mirror = tf.identity(trainingInputData*-1+1)

  #          for price_element_index in range(priceInputCount):
  #              one_hot_price_element_1d = tf.one_hot(price_element_index, priceInputCount, on_value= tf.cast(arrayindex, tf.int32), off_value = -1, dtype=tf.int32, name="price1d")
  #              #one_hot_price_element_1d = one_hot_price_element_1d * tf.cast(arrayindex+1, tf.int32) - 1
  #              one_hot_price_element_2d = tf.one_hot(one_hot_price_element_1d, max_batch_len, on_value=1, off_value = 0, dtype=tf.int32, axis=0, name="price2d")
  #              pricestensor = pricestensor + tf.cast(one_hot_price_element_2d, tf.float32) * trainingInputData[price_element_index]

  #              one_hot_price_element_mirror_1d = tf.one_hot(price_element_index, priceInputCount, on_value=tf.cast(arrayindex+1,tf.int32), off_value = -1, dtype=tf.int32, name="pricemirror1d")
  #              #one_hot_price_element_mirror_1d = one_hot_price_element_mirror_1d *tf.cast(arrayindex + 2,tf.int32) - 1
  #              one_hot_price_element_mirror_2d = tf.one_hot(one_hot_price_element_mirror_1d, max_batch_len, on_value=1, off_value = 0, dtype=tf.int32, axis=0, name="pricemirror2d")
  #              pricestensor = pricestensor + tf.cast(one_hot_price_element_mirror_2d, tf.float32) *  trainingInputData_Mirror[price_element_index]

  #          LabelData = labels[batchIndex][tf.cast(trainingIndex+priceInputCount-1,dtype=tf.int32)]
  #          LabelData = tf.stack([LabelData*-1+1, LabelData], axis=0)
  #          LabelData = tf.reshape(LabelData, [2])
  #          LabelData_Mirror = tf.identity(LabelData*-1+1)

  #          for label_element_index in range(2):
  #              one_hot_label_element_1d = tf.one_hot(label_element_index, 2, on_value=tf.cast(arrayindex,tf.int32), off_value = -1, dtype=tf.int32,name= "label1d")
  #              #one_hot_label_element_1d = one_hot_label_element_1d * tf.cast(arrayindex+1,tf.int32) - 1
  #              one_hot_label_element_2d = tf.one_hot(one_hot_label_element_1d, max_batch_len, on_value=1, off_value = 0, dtype=tf.int32, axis=0, name="label2d")
  #              labeltensor = labeltensor + tf.cast(one_hot_label_element_2d, tf.float32) * LabelData[label_element_index]
  #              one_hot_label_element_mirror_1d = tf.one_hot(label_element_index, 2, on_value=tf.cast(arrayindex+1,tf.int32), off_value = -1, dtype=tf.int32,name="labelmirror1d")
  #              #one_hot_label_element_mirror_1d = one_hot_label_element_mirror_1d * tf.cast(arrayindex+2,tf.int32) - 1
  #              one_hot_label_element_mirror_2d = tf.one_hot(one_hot_label_element_mirror_1d, max_batch_len, on_value=1, off_value = 0, dtype=tf.int32, axis=0,name="labelmirror2d")
  #              labeltensor = labeltensor + tf.cast(one_hot_label_element_mirror_2d, tf.float32) * LabelData_Mirror[label_element_index]

  #          trainingIndex = tf.subtract(trainingIndex, 1)
  #          arrayindex = tf.add(arrayindex, 2)
  #          return [arrayindex, trainingIndex, trainingCount, labeltensor, pricestensor]
  #        arrayindex, trainingIndex, trainingCount, labeltensor, pricestensor = tf.while_loop(while_cond, while_body, [arrayindex, trainingIndex, trainingCount, labeltensor, pricestensor], maximum_iterations=max_batch_len_tensor)
  #        return arrayindex, labeltensor, pricestensor

  #      def skip_training_set(arrayindex, labeltensor, pricestensor):
  #        return arrayindex, labeltensor, pricestensor
  #      arrayindex, labeltensor, pricestensor = tf.cond(tf.greater(labels_int[batchIndex],tf.constant(priceInputCount,dtype=tf.int64)),lambda: make_training_set(arrayindex, labeltensor, pricestensor),lambda: skip_training_set(arrayindex, labeltensor, pricestensor))

  #    original_index = tf.Variable(0, dtype=tf.int64, trainable=False)

  #    def while_cond_copy(original_index, arrayindex, labeltensor, pricestensor):
  #      return arrayindex < max_batch_len_tensor
  #    def while_body_copy(original_index, arrayindex, labeltensor, pricestensor):
  #      for price_element_index in range(priceInputCount):
  #        one_hot_1d = tf.one_hot(price_element_index, priceInputCount, on_value=tf.cast(arrayindex, tf.int32), off_value = -1, dtype=tf.int32)
  #        one_hot_2d = tf.one_hot(one_hot_1d, max_batch_len, on_value=1, off_value = 0, dtype=tf.int32, axis=0)
  #        pricestensor = pricestensor + tf.cast(one_hot_2d, tf.float32) * pricestensor[original_index][price_element_index]
  #      for label_element_index in range(2):
  #        one_hot_1d = tf.one_hot(label_element_index, 2, on_value=tf.cast(arrayindex, tf.int32), off_value = -1, dtype=tf.int32)
  #        one_hot_2d = tf.one_hot(one_hot_1d, max_batch_len, on_value=1, off_value = 0, dtype=tf.int32, axis=0)
  #        labeltensor = labeltensor + tf.cast(one_hot_2d, tf.float32) * labeltensor[original_index][label_element_index]
  #      arrayindex += 1
  #      original_index += 1
  #      return [original_index, arrayindex, labeltensor, pricestensor]
  #    original_index, arrayindex, labeltensor, pricestensor = tf.while_loop(while_cond_copy, while_body_copy, [original_index, arrayindex, labeltensor, pricestensor], maximum_iterations=max_batch_len_tensor)

  #    pricestensor = tf.reshape(pricestensor, [max_batch_len, PRICE_COUNT, DIMENSION_COUNT, CHANNEL_COUNT])

  #    if FLAGS.precision == 'bfloat16':
  #      with tf.tpu.bfloat16_scope():
  #        logits = build_network(pricestensor)
  #    elif FLAGS.precision == 'float32':
  #      logits = build_network(pricestensor)
        
  #if mode == tf.estimator.ModeKeys.PREDICT:
  if FLAGS.precision == 'bfloat16':
    with tf.tpu.bfloat16_scope():
      logits = build_network(features)
    logits = tf.cast(logits, tf.float32)
  elif FLAGS.precision == 'float32':
    logits = build_network(features)
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'classes': tf.argmax(logits, axis=1), 
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }
    
    tf.logging.info("classes=%s" % (tf.argmax(logits, axis=1)))
    tf.logging.info("probabilities=%s" % (predictions['probabilities']))
    
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })

  # If necessary, in the model_fn, use params['batch_size'] instead the batch
  # size flags (--train_batch_size or --eval_batch_size).
  batch_size = params['batch_size']   # pylint: disable=unused-variable
  
  tf.logging.info("logits=%s,labels=%s" % (logits.shape, labels.shape))
  #Calculate Loss: cross entropy
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits,
      #onehot_labels=labeltensor,
      onehot_labels=labels,
      label_smoothing=FLAGS.label_smoothing)

  #loss = cross_entropy + params['weight_decay'] * tf.add_n([
  loss = cross_entropy + FLAGS.weight_decay * tf.add_n([
        tf.nn.l2_loss(v)
        for v in tf.trainable_variables()
        if 'batch_normalization' not in v.name
    ])
    
  host_call = None
  if mode == tf.estimator.ModeKeys.TRAIN:
    # Compute the current epoch and associated learning rate from global_step.
    global_step = tf.train.get_global_step()
    steps_per_epoch = FLAGS.num_train_images / FLAGS.train_batch_size
    current_epoch = (tf.cast(global_step, tf.float32) /
                     steps_per_epoch)
    
    
    # LARS is a large batch optimizer. LARS enables higher accuracy at batch 16K
    # and larger batch sizes.

    '''
    if FLAGS.train_batch_size >= 16384 and FLAGS.enable_lars:
      learning_rate = 0.0
      optimizer = lars_util.init_lars_optimizer(current_epoch)
    else:
      learning_rate = learning_rate_schedule(FLAGS.train_steps, current_epoch)
      optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate,
          momentum=FLAGS.momentum,
          use_nesterov=True)

    '''

    # I think Adam optimizer is better than LARS/Momentum optimizer for ZeroAI
    # Boris Town 20190207
    optimizer = tf.train.AdamOptimizer()

    if FLAGS.use_tpu:
      # When using TPU, wrap the optimizer with CrossShardOptimizer which
      # handles synchronization details between different TPU cores. To the
      # user, this should look like regular synchronous training.
      #optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
      optimizer = tf.tpu.CrossShardOptimizer(optimizer)

    # Batch normalization requires UPDATE_OPS to be added as a dependency to
    # the train operation.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)

    if not FLAGS.skip_host_call:
      def host_call_fn(gs, loss):
        """Training host call. Creates scalar summaries for training metrics.

        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the
        model to the `metric_fn`, provide as part of the `host_call`. See
        https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
        for more information.

        Arguments should match the list of `Tensor` objects passed as the second
        element in the tuple passed to `host_call`.

        Args:
          gs: `Tensor with shape `[batch]` for the global_step
          loss: `Tensor` with shape `[batch]` for the training loss.
          lr: `Tensor` with shape `[batch]` for the learning_rate.
          ce: `Tensor` with shape `[batch]` for the current_epoch.

        Returns:
          List of summary ops to run on the CPU host.
        """
        gs = gs[0]
        # Host call fns are executed FLAGS.iterations_per_loop times after one
        # TPU loop is finished, setting max_queue value to the same as number of
        # iterations will make the summary writer only flush the data to storage
        # once per loop.
        with tf2.summary.create_file_writer(
            FLAGS.model_dir, 
            max_queue=FLAGS.iterations_per_loop).as_default():
          #with summary.always_record_summaries():
          with tf2.summary.record_if(True):
            tf2.summary.scalar('loss', loss[0], step=gs)
            #summary.scalar('learning_rate', lr[0], step=gs)
            #summary.scalar('current_epoch', ce[0], step=gs)

            return tf.summary.all_v2_summary_ops()

      # To log the loss, current learning rate, and epoch for Tensorboard, the
      # summary op needs to be run on the host CPU via host_call. host_call
      # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
      # dimension. These Tensors are implicitly concatenated to
      # [params['batch_size']].
      #learning_rate = 0.0
      gs_t = tf.reshape(global_step, [1])
      loss_t = tf.reshape(loss, [1])
      #lr_t = tf.reshape(learning_rate, [1])
      #ce_t = tf.reshape(current_epoch, [1])

      host_call = (host_call_fn, [gs_t, loss_t])

  else:
    train_op = None

  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:
    #def metric_fn(labels, labels_mirror, logits, logits_mirror):
    #def metric_fn(LabelsStack, logits):
    def metric_fn(labels, logits):
      """Evaluation metric function. Evaluates accuracy.

      This function is executed on the CPU and should not directly reference
      any Tensors in the rest of the `model_fn`. To pass Tensors from the model
      to the `metric_fn`, provide as part of the `eval_metrics`. See
      https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
      for more information.

      Arguments should match the list of `Tensor` objects passed as the second
      element in the tuple passed to `eval_metrics`.

      Args:
        labels: `Tensor` with shape `[batch]`.
        logits: `Tensor` with shape `[batch, num_classes]`.

      Returns:
        A dict of the metrics to return from evaluation.
      """

      k = 0
      labels = tf.argmax(labels, axis=1)
      predictions = tf.argmax(logits, axis=1)
      
      top_1_accuracys = tf.metrics.accuracy(labels, predictions)
      in_top_3 = tf.cast(tf.nn.in_top_k(logits, labels, 3), tf.float32)
      top_3_accuracys = tf.metrics.mean(in_top_3)
      in_top_6 = tf.cast(tf.nn.in_top_k(logits, labels, 6), tf.float32)
      top_6_accuracys = tf.metrics.mean(in_top_6)

      #top_1_accuracys = tf.metrics.mean(
      #  #tf.cast(tf.nn.in_top_k(tf.cast(LabelsStack,tf.float32), 
      #  tf.cast(tf.nn.in_top_k(tf.cast(labels,tf.float32), 
      #  predictions, 1), tf.float32))
      
      
      #top_3_accuracys = tf.metrics.mean(
      #  #tf.cast(tf.nn.in_top_k(tf.cast(LabelsStack,tf.float32), 
      #  tf.cast(tf.nn.in_top_k(tf.cast(labels,tf.float32), 
      #  predictions, 3), tf.float32))
      
      #top_6_accuracys = tf.metrics.mean(
      #  #tf.cast(tf.nn.in_top_k(tf.cast(LabelsStack,tf.float32), 
      #  tf.cast(tf.nn.in_top_k(tf.cast(labels,tf.float32), 
      #  predictions, 6), tf.float32))

      return {
          'Top-1-Accuracy': top_1_accuracys,
          'Top-3-Accuracy': top_3_accuracys,
          'Top-6-Accuracy': top_6_accuracys,
      }

    #eval_metrics = (metric_fn, [labeltensor, logits])
    eval_metrics = (metric_fn, [labels, logits])

  #return tf.contrib.tpu.TPUEstimatorSpec(
  return tf.estimator.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      host_call=host_call,
      eval_metrics=eval_metrics
  )

def _verify_non_empty_string(value, field_name):
  """Ensures that a given proposed field value is a non-empty string.

  Args:
    value:  proposed value for the field.
    field_name:  string name of the field, e.g. `project`.

  Returns:
    The given value, provided that it passed the checks.

  Raises:
    ValueError:  the value is not a string, or is a blank string.
  """
  if not isinstance(value, str):
    raise ValueError(
        'Bigtable parameter "%s" must be a string.' % field_name)
  if not value:
    raise ValueError(
        'Bigtable parameter "%s" must be non-empty.' % field_name)
  return value


def _select_tables_from_flags():
  """Construct training and evaluation Bigtable selections from flags.

  Returns:
    [training_selection, evaluation_selection]
  """
  project = _verify_non_empty_string(
      FLAGS.bigtable_project or FLAGS.gcp_project,
      'project')
  instance = _verify_non_empty_string(FLAGS.bigtable_instance, 'instance')
  table = _verify_non_empty_string(FLAGS.bigtable_table, 'table')
  train_prefix = _verify_non_empty_string(FLAGS.bigtable_train_prefix,
                                          'train_prefix')
  eval_prefix = _verify_non_empty_string(FLAGS.bigtable_eval_prefix,
                                         'eval_prefix')
  column_family = _verify_non_empty_string(FLAGS.bigtable_column_family,
                                           'column_family')
  column_qualifier = _verify_non_empty_string(FLAGS.bigtable_column_qualifier,
                                              'column_qualifier')
  return [
      imagenet_input.BigtableSelection(
          project=project,
          instance=instance,
          table=table,
          prefix=p,
          column_family=column_family,
          column_qualifier=column_qualifier)
      for p in (train_prefix, eval_prefix)
  ]


def main(unused_argv):
  if FLAGS.tpu or FLAGS.use_tpu: #Insert 20210214 BorisTown Turtle X 
      tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu if (FLAGS.tpu or FLAGS.use_tpu) else '',
          zone=FLAGS.tpu_zone,
          project=FLAGS.gcp_project)
  else:
      tpu_cluster_resolver = None

  if FLAGS.use_async_checkpointing:
    save_checkpoints_steps = None
  else:
    save_checkpoints_steps = max(100, FLAGS.iterations_per_loop)
  config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=save_checkpoints_steps,
      log_step_count_steps=FLAGS.log_step_count_steps,
      session_config=tf.ConfigProto(
          graph_options=tf.GraphOptions(
              rewrite_options=rewriter_config_pb2.RewriterConfig(
                  disable_meta_optimizer=True))),
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_cores,
          per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig
          .PER_HOST_V2))  # pylint: disable=line-too-long

  resnet_classifier = tf.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=resnet_model_fn,
      config=config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      #predict_batch_size=PREDICT_BATCH_SIZE,
      export_to_tpu=FLAGS.export_to_tpu)

  assert FLAGS.precision == 'bfloat16' or FLAGS.precision == 'float32', (
      'Invalid value for --precision flag; must be bfloat16 or float32.')
  tf.logging.info('Precision: %s', FLAGS.precision)
  use_bfloat16 = FLAGS.precision == 'bfloat16'

  # Input pipelines are slightly different (with regards to shuffling and
  # preprocessing) between training and evaluation.
  if FLAGS.bigtable_instance:
    tf.logging.info('Using Bigtable dataset, table %s', FLAGS.bigtable_table)
    select_train, select_eval = _select_tables_from_flags()
    
    imagenet_train, imagenet_eval = [imagenet_input.ImageNetBigtableInput(
        is_training=is_training,
        use_bfloat16=use_bfloat16,
        transpose_input=FLAGS.transpose_input,
        selection=selection) for (is_training, selection) in
                                     [(True, select_train),
                                      (False, select_eval)]]
    
  else:
    if FLAGS.data_dir == FAKE_DATA_DIR:
      tf.logging.info('Using fake dataset.')
    else:
      tf.logging.info('Using dataset: %s', FLAGS.data_dir)
    imagenet_train, imagenet_eval = [
        imagenet_input.ImageNetInput(
            is_training=is_training,
            data_dir=FLAGS.data_dir,
            prices_dir=FLAGS.prices_dir,
            predict_dir=FLAGS.predict_dir,
            transpose_input=FLAGS.transpose_input,
            cache=FLAGS.use_cache and is_training,
            price_count=PRICE_COUNT,
            num_parallel_calls=FLAGS.num_parallel_calls,
            use_bfloat16=use_bfloat16) for is_training in [True, False]
    ]

  steps_per_epoch = FLAGS.num_train_images // FLAGS.train_batch_size
  eval_steps = FLAGS.num_eval_images // FLAGS.eval_batch_size

  if FLAGS.mode == 'eval':

    # Run evaluation when there's a new checkpoint
    for ckpt in tf.train.checkpoints_iterator(
        FLAGS.model_dir, timeout=FLAGS.eval_timeout):
      tf.logging.info('Starting to evaluate.')
      try:
        start_timestamp = time.time()  # This time will include compilation time
        eval_results = resnet_classifier.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=eval_steps,
            checkpoint_path=ckpt)
        elapsed_time = int(time.time() - start_timestamp)
        tf.logging.info('Eval results: %s. Elapsed seconds: %d',
                        eval_results, elapsed_time)

        # Terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(ckpt).split('-')[1])
        if current_step >= FLAGS.train_steps:
          tf.logging.info(
              'Evaluation finished after training step %d', current_step)
          break

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        tf.logging.info(
            'Checkpoint %s no longer exists, skipping checkpoint', ckpt)

  else:   # FLAGS.mode == 'train' or FLAGS.mode == 'train_and_eval'
    try:
      current_step = tf.train.load_variable(FLAGS.model_dir,
                                            tf.GraphKeys.GLOBAL_STEP)
    except (TypeError, ValueError, tf.errors.NotFoundError):
      current_step = 0
    #current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)  # pylint: disable=protected-access,line-too-long
    tf.logging.info('model_dir=%s,steps=%d' % (FLAGS.model_dir,current_step))
    steps_per_epoch = FLAGS.num_train_images // FLAGS.train_batch_size

    tf.logging.info('Training for %d steps (%.2f epochs in total). Current'
                    ' step %d.',
                    FLAGS.train_steps,
                    FLAGS.train_steps / steps_per_epoch,
                    current_step)

    start_timestamp = time.time()  # This time will include compilation time

    if FLAGS.mode == 'train':
      hooks = []
      if FLAGS.use_async_checkpointing:
        try:
          from tensorflow.contrib.tpu.python.tpu import async_checkpoint  # pylint: disable=g-import-not-at-top
        except ImportError as e:
          logging.exception(
              'Async checkpointing is not supported in TensorFlow 2.x')
          raise e

        hooks.append(
            async_checkpoint.AsyncCheckpointSaverHook(
                checkpoint_dir=FLAGS.model_dir,
                save_steps=max(100, FLAGS.iterations_per_loop)))
      '''
      if FLAGS.profile_every_n_steps > 0:
        hooks.append(
            tpu_profiler_hook.TPUProfilerHook(
                save_steps=FLAGS.profile_every_n_steps,
                output_dir=FLAGS.model_dir, tpu=FLAGS.tpu)
            )
      '''
      resnet_classifier.train(
          input_fn=imagenet_train.input_fn,
          max_steps=FLAGS.train_steps,
          hooks=hooks)

    elif FLAGS.mode == 'train_and_eval':
      # assert FLAGS.mode == 'train_and_eval'
      while current_step < FLAGS.train_steps:
        # Train for up to steps_per_eval number of steps.
        # At the end of training, a checkpoint will be written to --model_dir.
        next_checkpoint = min(current_step + FLAGS.steps_per_eval,
                              FLAGS.train_steps)
        resnet_classifier.train(
            input_fn=imagenet_train.input_fn, max_steps=next_checkpoint)
        current_step = next_checkpoint

        tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                        next_checkpoint, int(time.time() - start_timestamp))

        # Evaluate the model on the most recent model in --model_dir.
        # Since evaluation happens in batches of --eval_batch_size, some images
        # may be excluded modulo the batch size. As long as the batch size is
        # consistent, the evaluated images are also consistent.
        tf.logging.info('Starting to evaluate.')
        eval_results = resnet_classifier.evaluate(
            input_fn= imagenet_eval.input_fn,
            steps=FLAGS.num_eval_images // FLAGS.eval_batch_size)
        tf.logging.info('Eval results at step %d: %s',
                        next_checkpoint, eval_results)

      elapsed_time = int(time.time() - start_timestamp)
      tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                      FLAGS.train_steps, elapsed_time)
    else: # FLAGS.mode == 'predict'
      
      price_file_pattern = os.path.join(
        FLAGS.prices_dir, 'price-*.csv')
      
      if FLAGS.mode == 'save_model':
        #20200817 save model
        resnet_classifier.export_saved_model(
          #export_dir_base='D:/saved_model/',
          export_dir_base=FLAGS.export_dir,
          serving_input_receiver_fn=imagenet_input.image_serving_input_fn
          )

      if FLAGS.mode == 'predict':
          while True:
            time.sleep(1)
            price_files  = glob.glob(price_file_pattern)
            if len(price_files) == 0:
              continue
            tf.logging.info('Starting to predict.')
            for price_file_item in price_files:
              with open(price_file_item,"r") as fcsv:
                csvreader = csv.reader(fcsv,delimiter = ",")
                price_batch_size = len(list(csvreader))
            
              # price_batch_size = PREDICT_BATCH_SIZE
          
              if price_batch_size == 0:
                continue
              #predictions = next(resnet_classifier.predict(
              #  input_fn=lambda params : imagenet_eval.predict_input_fn(params, price_batch_size),
              #  ), None)

              predictions = resnet_classifier.predict(
                input_fn=lambda params : imagenet_eval.predict_input_fn(params, price_batch_size, os.path.basename(price_file_item)),
                )
          
              tf.logging.info("predictions2 = %s" % predictions)
          
              # Output predictions to predict-0001.csv BorisTown 
              predict_filename_part = os.path.join(FLAGS.predict_dir, 'part-0001.part')
              predict_filename_csv = os.path.join(FLAGS.predict_dir, 'predict-0001.csv')
              if len(price_files) > 1:
                dirname = re.findall(r"price-(.+?)\.csv",price_file_item)[0]
                dirpath = os.path.join(FLAGS.predict_dir, dirname)
                if not os.path.exists(dirpath):
                  os.makedirs(dirpath)
                predict_filename_part = os.path.join(dirpath, 'part-0001.part')
                predict_filename_csv = os.path.join(dirpath, 'predict-0001.csv')
              predict_file = open(predict_filename_part, "w")
              predict_file.truncate()
              predict_line = ''
          
              #outarray = np.zeros([price_batch_size, MAX_CASE*LABEL_COUNT])
              outarray = np.zeros([price_batch_size, LABEL_COUNT])
          
              #for case_index, pred_item in enumerate(predictions):
              #for pred_item in enumerate(predictions):
                #tf.logging.info("pred_item_probabilities=%s" % (pred_item['probabilities']))
                #predict_line = ''
              #for batch_index, pred_operation in enumerate(predictions['probabilities']):
              for batch_index, pred_item in enumerate(predictions):
                pred_operation = pred_item['probabilities']
                #tf.logging.info("pred_operation.shape=%s" % (pred_operation.shape))
                for label_index in range(LABEL_COUNT):
                  #predict_line += str(pred_operation[k])
                  #tf.logging.info("prediction op:%s" % (pred_operation[label_index]))
                  #outarray[batch_index][case_index*LABEL_COUNT+label_index] = pred_operation[label_index]
                  outarray[batch_index][label_index] = pred_operation[label_index]
                   #predict_file.write(predict_line+'\n')
                #predict_file.close()
          
              #tf.logging.info('predict_line = %s' % (predict_line))
              for pred_row in outarray:
                predict_line = ''
                for pred_col in pred_row:
                  if predict_line != '':
                    predict_line += ','
                  predict_line += str(pred_col)
                predict_file.write(predict_line+'\n')
                tf.logging.info('%s' % (predict_line))
              predict_file.close()
              os.rename(predict_filename_part, predict_filename_csv)
              if(predict_line != ''):
                #for price_file in price_files:
                tf.logging.info('Removing ' + price_file_item)
                price_file_new = price_file_item.replace("price-", "backup-")
                os.rename(price_file_item, price_file_new)
            
    ##if FLAGS.export_dir is not None and FLAGS.mode != 'predict':
    #if FLAGS.export_dir is not None and FLAGS.mode == 'save_model':
    #  # The guide to serve a exported TensorFlow model is at:
    #  #    https://www.tensorflow.org/serving/serving_basic
    #  tf.logging.info('Starting to export model.')
    #  resnet_classifier.export_saved_model(
    #      export_dir_base=FLAGS.export_dir,
    #      serving_input_receiver_fn=imagenet_input.image_serving_input_fn)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
