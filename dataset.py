#!/usr/bin/python3
"""Holds the functions related to the dataset."""

import copy
import numpy as np
from ops import data_utilities
import tensorflow as tf

class Dataset():
    """Dataset holds a connectome dataset."""

    def __init__(self, filename):
        """initialize a dataset with train and test filenames."""
        dataset = np.load(filename)
        self.volume = dataset['volume'].astype(np.float32)
        self.label = dataset['label'].astype(np.uint8)
        self._converted_to_affinities = False
        self._sampled = False

    def save(self, filename = 'out.tfrecords'):
        """save the dataset to a tfrecords file."""
        with tf.python_io.TFRecordWriter(filename) as tfrecord_writer:
            image_count = 0
            for (vol, lab) in zip(self.volume, self.label):
                pass

    def split(self, percent = 0.8):
        """split the dataset at a percent."""
        idx = round(len(self.label)*percent)
        
        # create a new dataset
        new_dataset = copy.deepcopy(self)
        new_dataset.volume = self.volume[idx:]
        new_dataset.label = self.label[idx:]

        # update self
        self.volume = self.volume[:idx]
        self.label = self.label[:idx]

        return new_dataset

    def preprocess(self, preprocess):
        """preprocess the volumes."""
        pass

    def convert_to_affinities(self, affinity, long_range):
        """convert the label to affinities."""
        if self._converted_to_affinities:
            raise RuntimeError("already converted to affinities!")
        self.label = data_utilities.derive_affinities(
                affinity = affinity,
                label_volume = self.label,
                long_range = long_range)
        self._converted_to_affinities = True
    
    class DatasetSampler():
        """DatasetSampler generates the samples to conserve memory."""
        def __init__(self, dset, slices, stride):
            """initialize the DatasetSampler with a Dataset."""
            self.volume = dset.volume
            self.label = dset.label
            self.sample = iter(range(0, len(self.volume) - slices, stride))
            self.slices = slices

        def __iter__(self):
            """make this object an iterable."""
            return self

        def __next__(self):
            """next returns the next sample."""
            idx = next(self.sample)
            if self.slices == 1: # 2d dataset
                sampled_images = np.ndarray(self.volume.shape[1:] + (1,), dtype=np.float32)
                sampled_labels = np.ndarray(self.label.shape[1:], dtype=np.float32)
                sampled_images[:,:,0] = self.volume[idx:idx+self.slices]
                sampled_labels[:] = self.label[idx:idx+self.slices]
            else:
                sampled_images = np.ndarray((self.slices,) + self.volume.shape[1:] + (1,),
                        dtype=np.float32)
                sampled_labels = np.ndarray((self.slices,) + self.label.shape[1:],
                        dtype=np.float32)
                sampled_images[:,:,:,0] = self.volume[idx:idx+self.slices]
                sampled_labels[:] = self.label[idx:idx+self.slices]
            return sampled_images, sampled_labels

    def sampler(self, method = 'contiguous',
            slices = 1, stride = 1):
        """sample uses a method to sample from the labels/sets."""
        if method == 'contiguous':
            return self.DatasetSampler(self, slices, stride)
        else:
            raise RuntimeError("bad method")


