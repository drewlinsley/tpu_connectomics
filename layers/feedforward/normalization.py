import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import bfloat16

def batch(
        bottom,
        name,
        scale=True,
        center=True,
        training=True):
    print('batch: %s' % (str(bottom.dtype),))
    return tf.layers.batch_normalization(
        inputs=bottom,
        axis=len(bottom.shape)-1,
        name=name,
        scale=scale,
        center=center,
        training=training,
        fused=True)

