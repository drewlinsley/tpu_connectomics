import numpy as np
import tensorflow as tf
from scipy import sparse


def class_accuracy(logits, labels):
    """Accuracy of 1/n*sum(pred_i == label_i)."""
    return tf.reduce_mean(
        tf.to_float(
            tf.equal(
                tf.argmax(logits, 1),
                tf.squeeze(tf.cast(labels, dtype=tf.int64)))))


def adapted_rand(segB, segA, ravel=False):
    """Compute Adapted Rand error as defined as 1 - the maximal F-score of the
    Rand index.

    Implementation of the metric is taken from the CREMI contest:
    https://github.com/cremi/cremi_python/blob/master/cremi/evaluation/rand.py
    Input: seg - segmented mask
           gt - ground truth mask
    Output: are - A number between 0 and 1, lower number means better error
    """
    if ravel:
        segA = np.ravel(segA)
        segB = np.ravel(segB)
    n = segA.size

    n_labels_A = np.amax(segA) + 1
    n_labels_B = np.amax(segB) + 1

    ones_data = np.ones(n)

    p_ij = sparse.csr_matrix(
        (ones_data, (segA[:], segB[:])), shape=(n_labels_A, n_labels_B))

    a = p_ij[1:n_labels_A,:]
    b = p_ij[1:n_labels_A,1:n_labels_B]
    c = p_ij[1:n_labels_A,0].todense()
    d = b.multiply(b)

    a_i = np.array(a.sum(1))
    b_i = np.array(b.sum(0))

    sumA = np.sum(a_i * a_i)
    sumB = np.sum(b_i * b_i) + (np.sum(c) / n)
    sumAB = np.sum(d) + (np.sum(c) / n)

    precision = sumAB / sumB
    recall = sumAB / sumA

    fScore = 2.0 * precision * recall / (precision + recall)
    are = 1.0 - fScore

    return are.astype(np.float32)


def arand(seg, gt):
    """Compute Adapted Rand error as defined as 1 - the maximal F-score of 
    the Rand index. Using py_func inorder to work with tensors
    Input: seg - segmented mask
           gt - ground truth mask
    Output: are - A number between 0 and 1, lower number means better error
    """
    seg = tf.cast(tf.reshape(seg, [-1]), tf.int32)
    gt = tf.cast(tf.reshape(gt, [-1]), tf.int32)
    return tf.py_func(adapted_rand, [seg, gt], tf.float32)

