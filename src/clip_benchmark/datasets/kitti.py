# coding=utf-8
# Copyright 2019 Google LLC.
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

"""Implements Kitti data class."""

from __future__ import absolute_import, division, print_function

import numpy as np
import task_adaptation.data.base as base
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
from task_adaptation.registry import Registry


def _count_all_pp(x):
    """Count all objects."""
    # Count distribution (thresholded at 15):

    label = tf.math.minimum(tf.size(x['objects']['type']) - 1, 8)
    return {'image': x['image'], 'label': label}


def _count_vehicles_pp(x):
    """Counting vehicles."""
    # Label distribution:

    vehicles = tf.where(x['objects']['type'] < 3)  # Car, Van, Truck.
    # Cap at 3.
    label = tf.math.minimum(tf.size(vehicles), 3)
    return {'image': x['image'], 'label': label}


def _count_left_pp(x):
    """Count objects on the left hand side of the camera."""
    # Count distribution (thresholded at 15):

    # Location feature contains (x, y, z) in meters w.r.t. the camera.
    objects_on_left = tf.where(x['objects']['location'][:, 0] < 0)
    label = tf.math.minimum(tf.size(objects_on_left), 8)
    return {'image': x['image'], 'label': label}


def _count_far_pp(x):
    """Counts objects far from the camera."""
    # Threshold removes ~half of the objects.
    # Count distribution (thresholded at 15):

    # Location feature contains (x, y, z) in meters w.r.t. the camera.
    distant_objects = tf.where(x['objects']['location'][:, 2] >= 25)
    label = tf.math.minimum(tf.size(distant_objects), 8)
    return {'image': x['image'], 'label': label}


def _count_near_pp(x):
    """Counts objects close to the camera."""
    # Threshold removes ~half of the objects.
    # Count distribution:

    # Location feature contains (x, y, z) in meters w.r.t. the camera.
    close_objects = tf.where(x['objects']['location'][:, 2] < 25)
    label = tf.math.minimum(tf.size(close_objects), 8)
    return {'image': x['image'], 'label': label}


def _closest_object_distance_pp(x):
    """Predict the distance to the closest object."""
    # Label distribution:

    # Location feature contains (x, y, z) in meters w.r.t. the camera.
    dist = tf.reduce_min(x['objects']['location'][:, 2])
    thrs = np.array([-100, 5.6, 8.4, 13.4, 23.4])
    label = tf.reduce_max(tf.where((thrs - dist) < 0))
    return {'image': x['image'], 'label': label}


def _closest_vehicle_distance_pp(x):
    """Predict the distance to the closest vehicle."""
    # Label distribution:

    # Location feature contains (x, y, z) in meters w.r.t. the camera.
    vehicles = tf.where(x['objects']['type'] < 3)  # Car, Van, Truck.
    vehicle_z = tf.gather(params=x['objects']['location'][:, 2], indices=vehicles)
    vehicle_z = tf.concat([vehicle_z, tf.constant([[1000.0]])], axis=0)
    dist = tf.reduce_min(vehicle_z)
    # Results in a uniform distribution over three distances, plus one class for
    # "no vehicle".
    thrs = np.array([-100.0, 8.0, 20.0, 999.0])
    label = tf.reduce_max(tf.where((thrs - dist) < 0))
    return {'image': x['image'], 'label': label}


def _closest_object_x_location_pp(x):
    """Predict the absolute x position of the closest object."""
    # Label distribution:

    # Location feature contains (x, y, z) in meters w.r.t. the camera.
    idx = tf.math.argmin(x['objects']['location'][:, 2])
    xloc = x['objects']['location'][idx, 0]
    thrs = np.array([-100, -6.4, -3.5, 0.0, 3.3, 23.9])
    label = tf.reduce_max(tf.where((thrs - xloc) < 0))
    return {'image': x['image'], 'label': label}


_TASK_DICT = {
    'count_all': {
        'preprocess_fn': _count_all_pp,
        'num_classes': 16,
    },
    'count_left': {
        'preprocess_fn': _count_left_pp,
        'num_classes': 16,
    },
    'count_far': {
        'preprocess_fn': _count_far_pp,
        'num_classes': 16,
    },
    'count_near': {
        'preprocess_fn': _count_near_pp,
        'num_classes': 16,
    },
    'closest_object_distance': {
        'preprocess_fn': _closest_object_distance_pp,
        'num_classes': 5,
    },
    'closest_object_x_location': {
        'preprocess_fn': _closest_object_x_location_pp,
        'num_classes': 5,
    },
    'count_vehicles': {
        'preprocess_fn': _count_vehicles_pp,
        'num_classes': 4,
    },
    'closest_vehicle_distance': {
        'preprocess_fn': _closest_vehicle_distance_pp,
        'num_classes': 4,
    },
}


@Registry.register('data.kitti', 'class')
class KittiData(base.ImageTfdsData):
    """Provides Kitti dataset.

    Six tasks are supported:
      1. Count the number of objects.
      2. Count the number of objects on the left hand side of the camera.
      3. Count the number of objects in the foreground.
      4. Count the number of objects in the background.
      5. Predict the distance of the closest object.
      6. Predict the x-location (w.r.t. the camera) of the closest object.
    """

    def __init__(self, task, data_dir=None):
        if task not in _TASK_DICT:
            raise ValueError('Unknown task: %s' % task)

        dataset_builder = tfds.builder('kitti:3.3.0', data_dir=data_dir)
        dataset_builder.download_and_prepare()

        tfds_splits = {
            'train': 'train',
            'val': 'validation',
            'trainval': 'train+validation',
            'test': 'test',
            'train800': 'train[:800]',
            'val200': 'validation[:200]',
            'train800val200': 'train[:800]+validation[:200]',
        }

        # Example counts are retrieved from the tensorflow dataset info.
        train_count = dataset_builder.info.splits[tfds.Split.TRAIN].num_examples
        val_count = dataset_builder.info.splits[tfds.Split.VALIDATION].num_examples
        test_count = dataset_builder.info.splits[tfds.Split.TEST].num_examples
        # Creates a dict with example counts for each split.
        num_samples_splits = {
            'train': train_count,
            'val': val_count,
            'trainval': train_count + val_count,
            'test': test_count,
            'train800': 800,
            'val200': 200,
            'train800val200': 1000,
        }

        task = _TASK_DICT[task]
        base_preprocess_fn = task['preprocess_fn']
        super(KittiData, self).__init__(
            dataset_builder=dataset_builder,
            tfds_splits=tfds_splits,
            num_samples_splits=num_samples_splits,
            num_preprocessing_threads=400,
            shuffle_buffer_size=10000,
            base_preprocess_fn=base_preprocess_fn,
            num_classes=task['num_classes'],
        )
