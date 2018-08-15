#!/usr/bin/python3
"""runs a model on some data."""

from argparse import ArgumentParser
from config import Config
from dataset import Dataset
import os
import tensorflow as tf
from ops.data_utilities import apply_augmentations
import numpy as np

def main(config_filename, output_dirname):
    """run the model."""
    # get the configuration for the current run.
    config = Config(config_filename)

    train_dataset_path = '%s.npz' % (os.path.join(config['directory']['datasets'],
            config['train']['dataset']),)

    # get the training dataset and split it into training and testing sets.
    train_dataset = Dataset(train_dataset_path)

    # convert to affinities if needed
    if config['data']['convert_to_affinities']:
        train_dataset.convert_to_affinities(config['data']['affinity'],
                config['data']['long_range'])
    

    dims = [config['data']['z_slices']] + config['data']['input_shape']
    max_iter = 15
    for i in range(0,64):
        train_file = os.path.join(output_dirname, 'train-%d' % (i,))
        with tf.python_io.TFRecordWriter(train_file) as tfrecord_writer:
            print(i)
            for j in range(1,2):
                for (train_sample_volume, train_sample_label) in train_dataset.sampler(
                slices=config['data']['z_slices'],
                stride=config['data']['z_stride']):
                    train_sample_volume = np.expand_dims(train_sample_volume, 0)
                    train_sample_label = np.expand_dims(train_sample_label, 0)
                    train_sample_volume, train_sample_label = apply_augmentations(
                            train_sample_volume, train_sample_label, 
                            # [{k: v} for (k, v) in config['augmentations']],
                            [{'random_crop': []}], dims, dims)

                    # reshape for efficiency in the network
                    old_shape = train_sample_volume.shape
                    new_shape = (old_shape[2], old_shape[3], old_shape[1], old_shape[4])
                    train_sample_volume = np.reshape(train_sample_volume, new_shape)
                    
                    old_shape = train_sample_label.shape
                    new_shape = (old_shape[2], old_shape[3], old_shape[1], old_shape[4])
                    train_sample_label = np.reshape(train_sample_label, new_shape)

                    # for each run through the preprocessor -- 
                    #print(train_sample_volume.dtype)
                    #print(train_sample_label.dtype)
                    #print(train_sample_volume.shape)
                    #print(train_sample_label.shape)
        
                    example = tf.train.Example(
                            features=tf.train.Features(
                                feature={
                                    'volume': tf.train.Feature(
                                        float_list=tf.train.FloatList(
                                            value=train_sample_volume.flatten())),
                                    'label': tf.train.Feature(
                                        float_list=tf.train.FloatList(
                                            value=train_sample_label.flatten()))
                                    }))
                    tfrecord_writer.write(example.SerializeToString())

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
            '--config',
            dest='config_filename',
            default='configs/default.conf',
            help='Name of the configuration file')
    parser.add_argument(
            '--output',
            dest='output_dirname',
            default='gs://serrelab/c2/data/train.tfrecords',
            help='Output filename')
    args = parser.parse_args()
    main(**vars(args))
