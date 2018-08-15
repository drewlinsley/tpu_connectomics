#!/usr/bin/python3
"""runs a model on some data."""

#from config import Config
#from dataset import Dataset
#import os
import tensorflow as tf
#from tensorflow.contrib.tpu.python.tpu import bfloat16
#import time
#import flags
#import records
import models

def metric_fn(labels, logits):
    """calculate the accuracy"""
    accuracy = tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, axis=1))
    return {"accuracy": accuracy}

def fn_gen(model_name):
    """returns a model_fn for a given model."""

    build_model = models.model[model_name]
    def model_fn(features, labels, mode, params):
        """model function to give back a TPUEstimatorSpec."""
        use_tpu = params['use_tpu']
        train_logits = build_model(features, None, True, 3)
        # train_logits = build_model(features, None, True, 3)
        #train_logits = tf.cast(train_logits, tf.float32)
        print(features.shape)
        print(train_logits.shape)
        print(labels.shape)
        train_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=labels,
                    logits=train_logits))
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            # with bfloat16.bfloat16_scope():
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0003) #XXX:  change
            if params['use_tpu']:
                optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
            train_op = optimizer.minimize(
                loss=train_loss, global_step=tf.train.get_global_step())

            if params['use_tpu']:
                spec = tf.contrib.tpu.TPUEstimatorSpec(
                        mode=mode,
                        loss=train_loss,
                        train_op=train_op,
                        eval_metrics=(metric_fn, [labels, train_logits]))
            else:
                spec = tf.estimator.EstimatorSpec(
                        mode=tf.estimator.ModeKeys.TRAIN,
                        loss=train_loss,
                        train_op=train_op,
                        eval_metrics_ops={
                            "accuracy": ([labels, train_logits], metric_fn),
                        })
        if mode == tf.estimator.ModeKeys.EVAL:
            if params['use_tpu']:
                spec = tf.contrib.tpu.TPUEstimatorSpec(
                        mode=tf.estimator.ModeKeys.EVAL,
                        loss=train_loss,
                        eval_metrics=(metric_fn, [labels, train_logits]))
            else:
                spec = tf.estimator.EstimatorSpec(
                        mode=tf.estimator.ModeKeys.EVAL,
                        loss=train_loss)
#                        eval_metrics={
#                            "accuracy": ([labels, train_logits], metric_fn),
#                        })
        if mode == tf.estimator.ModeKeys.PREDICT:
            pass # predict the thing

        return spec

    return model_fn

