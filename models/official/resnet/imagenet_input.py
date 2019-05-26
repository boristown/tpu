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
# See the License for the specific language governing permissios and
# limitations under the License.
# ==============================================================================
"""Efficient ImageNet input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from collections import namedtuple
import functools
import os
import tensorflow as tf
import numpy as np
# from official.resnet import resnet_preprocessing
#import resnet_preprocessing

PRICE_COUNT = 10
DIMENSION_COUNT = 10
CHANNEL_COUNT = 1
LABEL_COUNT = 2
TEST_CASE = 1
MAX_CASE = 10

def image_serving_input_fn():
  """Serving input fn for raw images."""

  def _preprocess_image(image_bytes):
    """Preprocess a single raw image."""
    #image = resnet_preprocessing.preprocess_image(
    #    image_bytes=image_bytes, is_training=False)
    #return image
    return null

  image_bytes_list = tf.placeholder(
      shape=[None],
      dtype=tf.string,
  )
  images = tf.map_fn(
      _preprocess_image, image_bytes_list, back_prop=False, dtype=tf.float32)
  return tf.estimator.export.ServingInputReceiver(
      images, {'image_bytes': image_bytes_list})


class ImageNetTFExampleInput(object):
  """Base class for ImageNet input_fn generator.

  Args:
    is_training: `bool` for whether the input is for training
    use_bfloat16: If True, use bfloat16 precision; else use float32.
    transpose_input: 'bool' for whether to use the double transpose trick
    num_parallel_calls: `int` for the number of parallel threads.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self,
               is_training,
               use_bfloat16,
               price_count=PRICE_COUNT,
               transpose_input=False,
               num_parallel_calls=8):
    #raise Exception(f'ImageNetTFExampleInput init')
    #self.image_preprocessing_fn = resnet_preprocessing.preprocess_image
    self.is_training = is_training
    self.use_bfloat16 = use_bfloat16
    self.transpose_input = transpose_input
    self.prices_count = PRICE_COUNT
    self.num_parallel_calls = num_parallel_calls
    self.channelInputs = CHANNEL_COUNT
    self.operationOutputs = LABEL_COUNT

  def set_shapes(self, batch_size, prices, operations):
    #batch_real_size=batch_size*self.num_parallel_calls
    tf.logging.info("prices=%s,operations=%s" % (prices.shape,operations.shape))
    batch_real_size=batch_size
    """Statically set the batch_size dimension."""
    if self.transpose_input:
      prices.set_shape(prices.get_shape().merge_with(
          tf.TensorShape([None, None, None, batch_real_size])))
      prices = tf.reshape(prices, [-1])
      operations.set_shape(operations.get_shape().merge_with(
          tf.TensorShape([None, batch_real_size])))
      operations = tf.reshape(operations, [-1])
    else:
      prices.set_shape(prices.get_shape().merge_with(
          tf.TensorShape([batch_real_size, None, None, None])))
      operations.set_shape(operations.get_shape().merge_with(
          tf.TensorShape([batch_real_size, None])))
    tf.logging.info("prices=%s,operations=%s" % (prices.shape,operations.shape))
    return prices, operations

  def set_predict_shapes(self, batch_size, prices):
    tf.logging.info('prices.shape2=%s self.transpose_input=%s' % (prices.shape, self.transpose_input))
    """Statically set the batch_size dimension."""
    if self.transpose_input:
      prices.set_shape(prices.get_shape().merge_with(
          tf.TensorShape([None, None, None, batch_size])))
      prices = tf.reshape(prices, [-1])
    else:
      prices.set_shape(prices.get_shape().merge_with(
          tf.TensorShape([batch_size, None, None, None])))
    tf.logging.info('prices.shape3=%s' % (prices.shape) )
    return prices
  
  def dataset_parser(self, line):
    """Parses prices and its operations from a serialized ResNet-50 TFExample.

    Args:
      value: serialized string containing an ImageNet TFExample.

    Returns:
      Returns a tuple of (prices, operations) from the TFExample.
    """
    # Decode the csv_line to tensor.
    record_defaults = [[1.0] for col in range(PRICE_COUNT*DIMENSION_COUNT*CHANNEL_COUNT+LABEL_COUNT*MAX_CASE)]
    items = tf.decode_csv(line, record_defaults)
    prices = items[0:PRICE_COUNT*DIMENSION_COUNT*CHANNEL_COUNT]
    #prices = [0 if x==0.5 else x for x in prices]
    operations = items[PRICE_COUNT*DIMENSION_COUNT*CHANNEL_COUNT:PRICE_COUNT*DIMENSION_COUNT*CHANNEL_COUNT+LABEL_COUNT*MAX_CASE]
    if not self.use_bfloat16:
      prices = tf.cast(prices, tf.float32)
      operations = tf.cast(operations, tf.float32)
    else:
      prices = tf.cast(prices, tf.bfloat16)
      operations = tf.cast(operations, tf.bfloat16)
    prices = tf.reshape(prices,[PRICE_COUNT,DIMENSION_COUNT,CHANNEL_COUNT])
    tf.logging.info("prices=%s,operations=%s" % (prices.shape,operations.shape))
    return prices,operations

  def dataset_parser_tfrecord(self, line):
    """Parses prices and its operations from a serialized ResNet-50 TFExample.

    Args:
      value: serialized string containing an ImageNet TFExample.

    Returns:
      Returns a tuple of (prices, operations) from the TFExample.
    """
    keys_to_features = {
        'prices' : tf.FixedLenFeature([PRICE_COUNT*DIMENSION_COUNT*CHANNEL_COUNT], tf.float32, default_value=[0.0]*(PRICE_COUNT*DIMENSION_COUNT*CHANNEL_COUNT)),
        'label': tf.FixedLenFeature([1], tf.int64, default_value=[0]),
    }

    parsed = tf.parse_single_example(line, keys_to_features)
    
    prices = parsed['prices']
    operations = parsed['label']
    
    if not self.use_bfloat16:
      prices = tf.cast(prices, tf.float32)
      operations = tf.cast(operations, tf.int32)
    else:
      prices = tf.cast(prices, tf.bfloat16)
      operations = tf.cast(operations, tf.int32)
    prices = tf.reshape(prices,[PRICE_COUNT,DIMENSION_COUNT,CHANNEL_COUNT])
    operations = tf.reshape(operations, [1,1])
    #operations = tf.tile(operations, [MAX_CASE, LABEL_COUNT])
    #op_list = np.zeros([MAX_CASE, LABEL_COUNT])
    operationMod = tf.mod(operations, 2)
    operationDiv = tf.floordiv(operations, 2)
    labels = tf.identity(operationMod)
    for i in range(MAX_CASE-1):
        operationMod = tf.mod(operationDiv, 2)
        operationDiv = tf.floordiv(operationDiv, 2)
        #op_list[i][1] = parsed['label']
        labels = tf.concat([operationMod, labels], 0)
        #tf.logging.info("labels=%s" % (labels.shape))
    #operations = tf.cast(tf.stack(op_list), tf.int32)
    labels2 = tf.subtract(1, labels)
    labels = tf.concat([labels2, labels], 1)
    labels = tf.reshape(labels, [-1])
    tf.logging.info("prices=%s,labels=%s" % (prices.shape,labels.shape))
    return prices,labels
    
  def dataset_predict_parser(self, line):
    """Parses prices and its operations from a serialized ResNet-50 TFExample.
    Args:
      value: serialized string containing an ImageNet TFExample.

    Returns:
      Returns a tuple of (prices, operations) from the TFExample.
    """
    tf.logging.info('line=%s' % (line))
    # Decode the csv_line to tensor.
    record_defaults = [[1.0] for col in range(PRICE_COUNT*DIMENSION_COUNT*CHANNEL_COUNT)]
    items = tf.decode_csv(line, record_defaults)
    #tf.logging.info('items=%s' % (items))
    prices = items[0:PRICE_COUNT*DIMENSION_COUNT*CHANNEL_COUNT]
    #tf.logging.info('prices0=%s' % (prices))
    #prices = [0 if x==0.5 else x for x in prices]
    if not self.use_bfloat16:
      prices = tf.cast(prices, tf.float32)
    else:
      prices = tf.cast(prices, tf.bfloat16)
    prices = tf.reshape(prices,[PRICE_COUNT,DIMENSION_COUNT,CHANNEL_COUNT])
    
    tf.logging.info('prices.shape=%s' % (prices.shape))
    return prices
  
  @abc.abstractmethod
  def make_source_dataset_tfrecord(self, index, num_hosts):
    """Makes dataset of serialized TFExamples.

    The returned dataset will contain `tf.string` tensors, but these strings are
    serialized `TFExample` records that will be parsed by `dataset_parser`.

    If self.is_training, the dataset should be infinite.

    Args:
      index: current host index.
      num_hosts: total number of hosts.

    Returns:
      A `tf.data.Dataset` object. 
    """
    return

  def input_fn(self, params):
    """Input function which provides a single batch for train or eval.

    Args:
      params: `dict` of parameters passed from the `TPUEstimator`.
          `params['batch_size']` is always provided and should be used as the
          effective batch size.

    Returns:
      A `tf.data.Dataset` object.
    """
    #raise Exception(f'input_fn in class ImageNetTFExampleInput params:{params}')
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # tf.contrib.tpu.RunConfig for details.
    batch_size = params['batch_size']

    # TODO(dehao): Replace the following with params['context'].current_host
    if 'context' in params:
      current_host = params['context'].current_input_fn_deployment()[1]
      num_hosts = params['context'].num_hosts
    else:
      current_host = 0
      num_hosts = 1

    dataset = self.make_source_dataset_tfrecord(current_host, num_hosts)

    # Use the fused map-and-batch operation.
    #
    # For XLA, we must used fixed shapes. Because we repeat the source training
    # dataset indefinitely, we can use `drop_remainder=True` to get fixed-size
    # batches without dropping any training examples.
    #
    # When evaluating, `drop_remainder=True` prevents accidentally evaluating
    # the same image twice by dropping the final batch if it is less than a full
    # batch size. As long as this validation is done with consistent batch size,
    # exactly the same images will be used.
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            self.dataset_parser_tfrecord, batch_size=batch_size,
            num_parallel_batches=self.num_parallel_calls, drop_remainder=True))

    # Transpose for performance on TPU
    if self.transpose_input:
      dataset = dataset.map(
          lambda prices, operations: (tf.transpose(prices, [1, 2, 3, 0]), tf.transpose(operations, [1, 0])),
          num_parallel_calls=self.num_parallel_calls)

    # Assign static batch size dimension
    dataset = dataset.map(functools.partial(self.set_shapes, batch_size))

    # Prefetch overlaps in-feed with training
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    return dataset


  def predict_input_fn(self, params, batch_size, filepattern):
    """Input function which provides a single batch for predict.

    Args:
      params: `dict` of parameters passed from the `TPUEstimator`.
          `params['batch_size']` is always provided and should be used as the
          effective batch size.

    Returns:
      A `tf.data.Dataset` object.
    """
    #raise Exception(f'input_fn in class ImageNetTFExampleInput params:{params}')
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # tf.contrib.tpu.RunConfig for details.
    #batch_size = params['batch_size']
    #batch_size = 6

    # TODO(dehao): Replace the following with params['context'].current_host
    if 'context' in params:
      current_host = params['context'].current_input_fn_deployment()[1]
      num_hosts = params['context'].num_hosts
    else:
      current_host = 0
      num_hosts = 1
    tf.logging.info('current_host=%s num_hosts=%s batch_size=%s' % (current_host,num_hosts,batch_size))
    predict_dataset = self.make_predict_dataset(current_host, num_hosts, filepattern)

    # Use the fused map-and-batch operation.
    #
    # For XLA, we must used fixed shapes. Because we repeat the source training
    # dataset indefinitely, we can use `drop_remainder=True` to get fixed-size
    # batches without dropping any training examples.
    #
    # When evaluating, `drop_remainder=True` prevents accidentally evaluating
    # the same image twice by dropping the final batch if it is less than a full
    # batch size. As long as this validation is done with consistent batch size,
    # exactly the same images will be used.
    predict_dataset = predict_dataset.apply(
#    tf.contrib.data.map(self.dataset_parser))
        tf.contrib.data.map_and_batch(
            self.dataset_predict_parser, batch_size=batch_size,
            num_parallel_batches=self.num_parallel_calls, drop_remainder=True))

    # Transpose for performance on TPU
    if self.transpose_input:
      predict_dataset = predict_dataset.map(
         lambda prices: tf.transpose(prices, [1, 2, 3, 0]),
         num_parallel_calls=self.num_parallel_calls)

    # Assign static batch size dimension
    predict_dataset = predict_dataset.map(functools.partial(self.set_predict_shapes, batch_size))

    # Prefetch overlaps in-feed with training
    predict_dataset = predict_dataset.prefetch(tf.contrib.data.AUTOTUNE)
    tf.logging.info('predict_dataset=%s' % (predict_dataset))
    return predict_dataset


class ImageNetInput(ImageNetTFExampleInput):
  """Generates ImageNet input_fn from a series of TFRecord files.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:

      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  The validation data is in the same format but sharded in 128 files.

  The format of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py
  """

  def __init__(self,
               is_training,
               use_bfloat16,
               transpose_input,
               data_dir,
               prices_dir,
               predict_dir,
               price_count=PRICE_COUNT,
               num_parallel_calls=8,
               cache=False):
    """Create an input from TFRecord files.

    Args:
      is_training: `bool` for whether the input is for training
      use_bfloat16: If True, use bfloat16 precision; else use float32.
      transpose_input: 'bool' for whether to use the double transpose trick
      data_dir: `str` for the directory of the training and validation data;
          if 'null' (the literal string 'null') or implicitly False
          then construct a null pipeline, consisting of empty images
          and blank labels.
      image_size: `int` image height and width.
      num_parallel_calls: concurrency level to use when reading data from disk.
      cache: if true, fill the dataset by repeating from its cache
    """
    super(ImageNetInput, self).__init__(
        is_training=is_training,
        price_count=price_count,
        use_bfloat16=use_bfloat16,
        transpose_input=transpose_input)
    self.data_dir = data_dir
    self.predict_dir = predict_dir
    self.prices_dir = prices_dir
    # TODO(b/112427086):  simplify the choice of input source
    if self.data_dir == 'null' or not self.data_dir:
      self.data_dir = None
    self.num_parallel_calls = num_parallel_calls
    self.cache = cache

  def _get_null_input(self, data):
    """Returns a null image (all black pixels).

    Args:
      data: element of a dataset, ignored in this method, since it produces
          the same null image regardless of the element.

    Returns:
      a tensor representing a null image.
    """
    del data  # Unused since output is constant regardless of input
    return tf.zeros([PRICE_COUNT, DIMENSION_COUNT, CHANNEL_COUNT], tf.bfloat16
                    if self.use_bfloat16 else tf.float32)

  def dataset_parser(self, value):
    """See base class."""
    #raise Exception('This is dataset_parser in class ImageNetInput')
    if not self.data_dir:
      return value, tf.constant(0, tf.int32)
    return super(ImageNetInput, self).dataset_parser(value)
    
  def dataset_parser_tfrecord(self, value):
    """See base class."""
    #raise Exception('This is dataset_parser in class ImageNetInput')
    if not self.data_dir:
      return value, tf.constant(0, tf.int32)
    return super(ImageNetInput, self).dataset_parser_tfrecord(value)

  def dataset_predict_parser(self, value):
    """See base class."""
    assert len(self.predict_dir) > 0
    if not self.predict_dir:
      return value
    return super(ImageNetInput, self).dataset_predict_parser(value)
  
  def make_source_dataset_tfrecord(self, index, num_hosts):
    """See base class."""
    if not self.data_dir:
      tf.logging.info('Undefined data_dir implies null input')
      return tf.data.Dataset.range(1).repeat().map(self._get_null_input)

    # Shuffle the filenames to ensure better randomization.
    file_pattern = os.path.join(
      self.data_dir, 'train-*' if self.is_training else 'validation-*')
    #  self.data_dir, 'train-*')
    #raise Exception(f'file_pattern = {filename} in class ImageNetInput')
    # For multi-host training, we want each hosts to always process the same
    # subset of files.  Each host only sees a subset of the entire dataset,
    # allowing us to cache larger datasets in memory.
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
    dataset = dataset.shard(num_hosts, index)

    if self.is_training and not self.cache:
      dataset = dataset.repeat()

    def fetch_dataset(filename):
      #raise Exception(f'fetch_dataset {filename} in class ImageNetInput')
      tf.logging.info("filename.shape = %s" % (filename.shape))
      buffer_size = 32 * 1024 * 1024  # 32 MiB per file
      dataset = tf.data.TFRecordDataset(filename, compression_type="ZLIB", buffer_size=buffer_size)
      #dataset = tf.data.TextLineDataset(filename, buffer_size=buffer_size)
      return dataset

    # Read the data from disk in parallel
    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            fetch_dataset, cycle_length=64, sloppy=True))

    if self.cache:
      dataset = dataset.cache().apply(
          tf.contrib.data.shuffle_and_repeat(1024 * 32))
    else:
      dataset = dataset.shuffle(1024)
    return dataset

  def make_predict_dataset(self, index, num_hosts, filepattern):
    """See base class."""
    if not self.prices_dir:
      tf.logging.info('Undefined prices_dir implies null input')
      return tf.data.Dataset.range(1).repeat().map(self._get_null_input)

    # Shuffle the filenames to ensure better randomization.
    price_file_pattern = os.path.join(
      #self.prices_dir, 'price-*')
      self.prices_dir, filepattern)
    tf.logging.info('price_file_pattern = %s index = %s' % (price_file_pattern,index))
    # For multi-host training, we want each hosts to always process the same
    # subset of files.  Each host only sees a subset of the entire dataset,
    # allowing us to cache larger datasets in memory.
    predict_dataset = tf.data.Dataset.list_files(price_file_pattern, shuffle=False)
    tf.logging.info('predict_dataset1=%s' % (predict_dataset))
    predict_dataset = predict_dataset.shard(num_hosts, index)
    tf.logging.info('predict_dataset2=%s' % (predict_dataset))

    #if self.is_training and not self.cache:
    #  dataset = dataset.repeat()

    def fetch_predict_dataset(filename):
      #raise Exception(f'fetch_dataset {filename} in class ImageNetInput')
      tf.logging.info("filename.shape = %s" % (filename.shape))
      buffer_size = 32 * 1024 * 1024  # 32 MiB per file
      predict_dataset = tf.data.TextLineDataset(filename, buffer_size=buffer_size)
      #tf.logging.info("predict_dataset1 = %s" % predict_dataset.shapes)
      return predict_dataset

    # Read the data from disk in parallel
    predict_dataset = predict_dataset.apply(
        tf.contrib.data.parallel_interleave(
            fetch_predict_dataset, cycle_length=64, sloppy=True))
    '''
    if self.cache:
      predict_dataset = predict_dataset.cache().apply(
          tf.contrib.data.shuffle_and_repeat(1024 * 16))
    else:
      predict_dataset = dataset.shuffle(1024)
    '''

    tf.logging.info("predict_dataset = %s" % predict_dataset)
    return predict_dataset
  
# Defines a selection of data from a Cloud Bigtable.
BigtableSelection = namedtuple('BigtableSelection',
                               ['project',
                                'instance',
                                'table',
                                'prefix',
                                'column_family',
                                'column_qualifier'])


class ImageNetBigtableInput(ImageNetTFExampleInput):
  """Generates ImageNet input_fn from a Bigtable for training or evaluation.
  """

  def __init__(self, is_training, use_bfloat16, transpose_input, selection):
    """Constructs an ImageNet input from a BigtableSelection.

    Args:
      is_training: `bool` for whether the input is for training
      use_bfloat16: If True, use bfloat16 precision; else use float32.
      transpose_input: 'bool' for whether to use the double transpose trick
      selection: a BigtableSelection specifying a part of a Bigtable.
    """
    super(ImageNetBigtableInput, self).__init__(
        is_training=is_training,
        use_bfloat16=use_bfloat16,
        transpose_input=transpose_input)
    self.selection = selection

  def make_source_dataset(self, index, num_hosts):
    """See base class."""
    data = self.selection
    client = tf.contrib.cloud.BigtableClient(data.project, data.instance)
    table = client.table(data.table)
    ds = table.parallel_scan_prefix(data.prefix,
                                    columns=[(data.column_family,
                                              data.column_qualifier)])
    # The Bigtable datasets will have the shape (row_key, data)
    ds_data = ds.map(lambda index, data: data)

    if self.is_training:
      ds_data = ds_data.repeat()

    return ds_data
