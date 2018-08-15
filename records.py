#!/usr/bin/python3

import functools
import os
import tensorflow as tf

class Reader():
    """Reader reads from a tfrecord file to produce an image."""

    def __init__(self, data_dir, dims, channels):
        """initialize the reader with a tfrecord dir and dims."""
        self.data_dir = data_dir
        self.dims = dims
        self.channels = channels

    def dataset_parser(self, value):
        """parse the tfrecords."""
        keys_to_features = {
                'volume': tf.FixedLenFeature(self.dims + [1], tf.float32),
                'label': tf.FixedLenFeature(self.dims + [self.channels], tf.float32)
            }
        
        parsed = tf.parse_single_example(value, keys_to_features)
        print(parsed['volume'].shape)
        print(parsed['label'].shape)
        volume = parsed['volume']
        label = parsed['label']
        #volume = tf.cast(volume, tf.bfloat16)
        #label = tf.cast(label, tf.bfloat16)
        # volume = tf.reshape(parsed['volume'], shape=self.dims)
        #label = tf.cast(
        #        tf.reshape(parsed['label'], shape=self.dims), dtype=tf.int32)
        
        return volume, label

    def set_shapes(self, batch_size, volume, label):
        '''Statically set the size of each component.'''
        print(volume.shape)
        volume.set_shape(volume.get_shape().merge_with(
            tf.TensorShape([batch_size])))
        print(volume.shape)
        label.set_shape(label.get_shape().merge_with(
            tf.TensorShape([batch_size])))

    def input_fn(self, params):
        """input function provides a single batch for train or eval."""
        batch_size = params['batch_size']
        is_training = params['train']
        print(batch_size)

        file_pattern = os.path.join(
                self.data_dir, 'train-*' if is_training else 'validation-*')
        dataset = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)
        
        if is_training:
            dataset = dataset.repeat()

        def fetch_dataset(filename):
            buffer_size = 1024*1024*8
            return tf.data.TFRecordDataset(filename, buffer_size=buffer_size)

        dataset = dataset.apply(
                tf.contrib.data.parallel_interleave(
                    fetch_dataset, cycle_length=64, sloppy=True))
        dataset = dataset.shuffle(1024)
        
        dataset = dataset.apply(
                tf.contrib.data.map_and_batch(
                    self.dataset_parser, batch_size=batch_size,
                    num_parallel_batches=8, drop_remainder=True))
        
        # XXX: static batch size?
        # dataset = dataset.map(functools.partial(self.set_shapes, batch_size))

        dataset = dataset.prefetch(32)
        
        return dataset
