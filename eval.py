#!/usr/bin/python3
"""runs a model on some data."""

from config import Config
from dataset import Dataset
import os
import tensorflow as tf
#from tensorflow.contrib.tpu.python.tpu import bfloat16
import time
import flags
import records
import models
import model

def main(config_filename = 'configs/default3d.conf'):
    """run the model."""

    # get the configuration for the current run.
    config = Config(config_filename)
    FLAGS = flags.get()

    # XXX: grab this from config
    datareader = records.Reader(FLAGS.data_dir,
            config['data']['input_shape'] + [config['data']['z_slices']],
            config['data']['affinity'])

    tf.logging.set_verbosity(tf.logging.INFO)
    
    print(FLAGS.use_tpu)
    if FLAGS.use_tpu:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                FLAGS.tpu,
                zone=FLAGS.tpu_zone,
                project=FLAGS.gcp_project)

        run_config = tf.contrib.tpu.RunConfig(
                cluster=tpu_cluster_resolver,
                model_dir=FLAGS.model_dir,
                session_config=tf.ConfigProto(
                    allow_soft_placement=False,
                    log_device_placement=True),
                tpu_config=tf.contrib.tpu.TPUConfig(
                    iterations_per_loop=FLAGS.iterations,
                    num_shards=FLAGS.num_shards))

        estimator = tf.contrib.tpu.TPUEstimator(
                model_fn=model.fn_gen(config['network']['model']),
                use_tpu=FLAGS.use_tpu,
                train_batch_size=FLAGS.batch_size,
                eval_batch_size=FLAGS.batch_size,
                params={"data_dir": FLAGS.data_dir,
                    "train": True,
                    'use_tpu': FLAGS.use_tpu},
                config=run_config)
    else:
        run_config = tf.estimator.RunConfig().replace(
		session_config=tf.ConfigProto(log_device_placement=True,
			device_count={'GPU': 1}))
        estimator = tf.estimator.Estimator(
                config=run_config,
                model_fn=model.fn_gen(config['network']['model']),
                model_dir=FLAGS.model_dir,
                params={"data_dir": FLAGS.data_dir,
                    "train": False,
                    'use_tpu': FLAGS.use_tpu,
                    'batch_size': FLAGS.batch_size},
                )


    start_time = time.time()
    estimator.evaluate(input_fn=datareader.input_fn)

    elapsed_time = int(time.time() - start_time)
    tf.logging.info('Finished evaluation up to step %d in %d seconds.' % (FLAGS.train_steps, elapsed_time))

    
if __name__ == '__main__':
    main()
