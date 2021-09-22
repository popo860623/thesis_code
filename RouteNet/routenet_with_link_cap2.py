from __future__ import print_function

import numpy as np
import pandas as pd
import networkx as nx
import itertools as it
import os
import tensorflow as tf
from tensorflow import keras
import re
import argparse
import random
import tarfile


def make_indices(paths):
    # print('paths = ', paths)
    link_indices = []
    path_indices = []
    sequ_indices = []
    segment = 0
    for p in paths:
        link_indices += p
        path_indices += len(p)*[segment]
        sequ_indices += list(range(len(p)))
        segment += 1
    return link_indices, path_indices, sequ_indices


def _int64_feature(value):
    return tf.compat.v1.train.Feature(int64_list=tf.compat.v1.train.Int64List(value=[value]))


def _int64_features(value):
    return tf.compat.v1.train.Feature(int64_list=tf.compat.v1.train.Int64List(value=value))


def _float_features(value):
    return tf.compat.v1.train.Feature(float_list=tf.compat.v1.train.FloatList(value=value))


def parse(serialized, target='delay'):
    '''
    Target is the name of predicted variable
    '''
    with tf.device("/cpu:0"):
        with tf.name_scope('parse'):
            features = tf.io.parse_single_example(
                serialized,
                features={
                    'traffic': tf.io.VarLenFeature(tf.float32),
                    target: tf.io.VarLenFeature(tf.float32),
                    'link_capacity': tf.io.VarLenFeature(tf.float32),
                    'links': tf.io.VarLenFeature(tf.int64),
                    'paths': tf.io.VarLenFeature(tf.int64),
                    'sequences': tf.io.VarLenFeature(tf.int64),
                    'n_links': tf.io.FixedLenFeature([], tf.int64),
                    'n_paths': tf.io.FixedLenFeature([], tf.int64),
                    'n_total': tf.io.FixedLenFeature([], tf.int64)
                })

            # Normalization
            for k in ['traffic', target, 'link_capacity', 'links', 'paths', 'sequences']:
                features[k] = tf.sparse.to_dense(features[k])
  
                # if k == 'traffic':
                #     features[k] = (features[k]) / 58
                # if k == 'link_capacity':
                #     features[k] = (features[k] ) / 40000
                # tf.print('==================================')

    # print ('features[target] = ', tf.print(features[target]) )

    return {k: v for k, v in features.items() if k is not target}, features[target]


def cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]

        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes)-1):
            cummaxes.append(tf.math.add_n(maxes[0:i+1]))

    return cummaxes


def transformation_func(it, batch_size=8):
    with tf.name_scope("transformation_func"):
        vs = [it.get_next() for _ in range(batch_size)]
        # with tf.Session() as sess:
        #     print('+++', sess.run(vs))
        # print('vs = ',vs)
        # print('transformation_func vs = %s' % vs)
        links_cummax = cummax(vs, lambda v: v[0]['links'])
        paths_cummax = cummax(vs, lambda v: v[0]['paths'])
        # print('paths = ', vs[0]['paths'])
        # print('************************************')
        tensors = ({
            'traffic': tf.concat([v[0]['traffic'] for v in vs], axis=0),
            'sequences': tf.concat([v[0]['sequences'] for v in vs], axis=0),
            'link_capacity': tf.concat([v[0]['link_capacity'] for v in vs], axis=0),
            'links': tf.concat([v[0]['links'] + m for v, m in zip(vs, links_cummax)], axis=0),
            'paths': tf.concat([v[0]['paths'] + m for v, m in zip(vs, paths_cummax)], axis=0),
            'n_links': tf.math.add_n([v[0]['n_links'] for v in vs]),
            'n_paths': tf.math.add_n([v[0]['n_paths'] for v in vs]),
            'n_total': tf.math.add_n([v[0]['n_total'] for v in vs])
        },   tf.concat([v[1] for v in vs], axis=0))

    return tensors


def tfrecord_input_fn(filenames, hparams, shuffle_buf=1000, target='delay'):
    # print('filename = ', filenames)
    # If your input data is stored in a file in the recommended TFRecord format
    files = tf.data.Dataset.from_tensor_slices(filenames)
    # filenames = arg.train = train's tfrecord file
    files = files.shuffle(len(filenames))
    # print('files = ', files)
    ds = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=4))

    if shuffle_buf:
        ds = ds.apply(tf.contrib.data.shuffle_and_repeat(shuffle_buf))

    ds = ds.map(lambda buf: parse(buf, target),
                num_parallel_calls=2)

    print('after parse : %s' % str(ds))
    ds = ds.prefetch(10)

    it = ds.make_one_shot_iterator()
    sample = transformation_func(it)
    # print('sample = ', sample)
    return sample
# self define model


class ComnetModel(tf.keras.Model):
    def __init__(self, hparams, output_units=1, final_activation=None):
        super(ComnetModel, self).__init__()
        self.hparams = hparams

        self.edge_update = tf.keras.layers.GRUCell(hparams.link_state_dim)
        self.path_update = tf.keras.layers.GRUCell(hparams.path_state_dim)

        self.readout = tf.keras.models.Sequential()

        self.readout.add(keras.layers.Dense(hparams.readout_units, activation=tf.nn.leaky_relu,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(hparams.l2),
                                            kernel_initializer='he_normal'
                                            ))
        # self.readout.add(keras.layers.Dense(hparams.readout_units, activation=tf.nn.selu,
        #                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(hparams.l2),
        #                                     kernel_initializer='lecun_normal'))
        self.readout.add(keras.layers.Dropout(hparams.dropout_rate))
        self.readout.add(keras.layers.Dense(hparams.readout_units, activation=tf.nn.leaky_relu,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(hparams.l2),
                                            kernel_initializer='he_normal'
                                            ))
        self.readout.add(keras.layers.Dropout(hparams.dropout_rate))# hparams.dropout_rate
        # self.readout.add(keras.layers.Dense(hparams.readout_units, activation=tf.nn.leaky_relu,
        #                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(hparams.l2),
        #                                     # kernel_initializer='he_normal'
        #                                     ))
        # self.readout.add(keras.layers.Dropout(hparams.dropout_rate))
        self.readout.add(keras.layers.Dense(output_units, kernel_regularizer=tf.contrib.layers.l2_regularizer(
            hparams.l2_2), activation=final_activation))

    # input shape , define variable
    def build(self, input_shape=None):
        del input_shape
        self.edge_update.build(tf.TensorShape(
            [None, self.hparams.path_state_dim]))  # No batch
        self.path_update.build(tf.TensorShape(
            [None, self.hparams.link_state_dim]))  # No batch

        self.readout.build(input_shape=[None, self.hparams.path_state_dim])
        self.built = True

    # feed forward
    def call(self, inputs, training=False):
        f_ = inputs
        # print('inputs = ', f_)
        shape = tf.stack(
            [f_['n_links'], self.hparams.link_state_dim-1], axis=0)
        link_state = tf.concat([
            tf.expand_dims(f_['link_capacity'], axis=1),
            tf.zeros(shape)
        ], axis=1)
        # print('link_shape = ', tf.shape(link_state))

        shape = tf.stack(
            [f_['n_paths'], self.hparams.path_state_dim-1], axis=0)
        path_state = tf.concat([
            tf.expand_dims(f_['traffic'][0:f_["n_paths"]], axis=1),
            tf.zeros(shape)
        ], axis=1)
        
        links = f_['links']
        paths = f_['paths']
        seqs = f_['sequences']

        for _ in range(self.hparams.T):

            h_tild = tf.gather(link_state, links)

            ids = tf.stack([paths, seqs], axis=1)
            max_len = tf.reduce_max(seqs)+1
            shape = tf.stack(
                [f_['n_paths'], max_len, self.hparams.link_state_dim])
            lens = tf.math.segment_sum(data=tf.ones_like(paths),
                                       segment_ids=paths)

            link_inputs = tf.scatter_nd(ids, h_tild, shape)
            outputs, path_state = tf.nn.dynamic_rnn(self.path_update,          # Line 10
                                                    link_inputs,
                                                    sequence_length=lens,
                                                    initial_state=path_state,
                                                    dtype=tf.float32)
            m = tf.gather_nd(outputs, ids)   # Line 11
            m = tf.math.unsorted_segment_sum(
                m, links, f_['n_links'])   # Line 16

            # Keras cell expects a list
            link_state, _ = self.edge_update(m, [link_state])  # Line 17

        if self.hparams.learn_embedding:
            r = self.readout(path_state, training=training)
        # else:
        #     r = self.readout(tf.stop_gradient(path_state),training=training)

        return r


def model_fn(
        features,  # This is batch_features from input_fn
        labels,   # This is batch_labrange
        mode,     # An instance of tf.estimator.ModeKeys
        params):  # Additional configuration

    model = ComnetModel(params)
    model.build()
    model.summary()

    def fn(x):
        r = model(x, training=mode == tf.estimator.ModeKeys.TRAIN)
        return r

    predictions = fn(features)

    predictions = tf.squeeze(predictions)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,
                                          predictions={'predictions': predictions})

    loss = tf.losses.mean_squared_error(
        labels=labels,
        predictions=predictions,
        reduction=tf.compat.v1.losses.Reduction.MEAN
    )
    # loss = tf.keras.losses.MSE(labels,predictions)
    regularization_loss = sum(model.losses)
    total_loss = loss + regularization_loss
    # 透過 tf.summary 在原本的 Graph 上面把想要記錄的項目加進 TensorBoard 之中
    tf.compat.v1.summary.scalar('loss', loss)
    tf.compat.v1.summary.scalar('regularization_loss', regularization_loss)
    
    if mode == tf.estimator.ModeKeys.EVAL:
        print('labels = ',labels)
        print('pred = ',predictions)
        print('eval loss = ', loss)
        return tf.estimator.EstimatorSpec(
            mode, loss=loss,
            eval_metric_ops={
                'label/mean': tf.metrics.mean(labels),
                'prediction/mean': tf.metrics.mean(predictions),
                'mae': tf.metrics.mean_absolute_error(labels, predictions),
                'rho': tf.contrib.metrics.streaming_pearson_correlation(labels=labels, predictions=predictions),
                'mre': tf.metrics.mean_relative_error(labels, predictions, labels),
                'r-squared': r_squared(labels, predictions)
            }
        )

    assert mode == tf.estimator.ModeKeys.TRAIN

    trainables = model.variables
    grads = tf.gradients(total_loss, trainables)
    grad_var_pairs = zip(grads, trainables)

    summaries = [tf.compat.v1.summary.histogram(
        var.op.name, var) for var in trainables]
    summaries += [tf.compat.v1.summary.histogram(g.op.name, g)
                  for g in grads if g is not None]

    decayed_lr = tf.compat.v1.train.exponential_decay(params.learning_rate,
                                                      tf.compat.v1.train.get_global_step(), 10000,
                                                      0.9, staircase=True)

    optimizer = tf.compat.v1.train.AdamOptimizer(decayed_lr)
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(grad_var_pairs,
                                             global_step=tf.compat.v1.train.get_global_step())

    logging_hook = tf.estimator.LoggingTensorHook(
        {"Loss": loss,
         "Regularization loss": regularization_loss,
         "Total loss": total_loss}, every_n_iter=10)

    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      training_hooks=[logging_hook]
                                      )


hparams = tf.contrib.training.HParams(
    link_state_dim=128,
    path_state_dim=128,
    T=4,
    readout_units=512,
    learning_rate=0.0001,
    batch_size=32,
    dropout_rate=0.5,
    l2=0.01,
    l2_2=0.01,
    learn_embedding=True  # If false, only the readout is trained
)


def r_squared(labels, predictions):
    """Computes the R^2 score.

        Args:
            labels (tf.Tensor): True values
            labels (tf.Tensor): This is the second item returned from the input_fn passed to train, evaluate, and predict.
                                If mode is tf.estimator.ModeKeys.PREDICT, labels=None will be passed.

        Returns:
            tf.Tensor: Mean R^2
        """

    total_error = tf.reduce_sum(tf.square(labels - tf.reduce_mean(labels)))
    unexplained_error = tf.reduce_sum(tf.square(labels - predictions))
    r_sq = 1.0 - tf.truediv(unexplained_error, total_error)

    # Needed for tf2 compatibility.
    m_r_sq, update_rsq_op = tf.compat.v1.metrics.mean(r_sq)

    return m_r_sq, update_rsq_op


def train(args):
    print('args = \n %s ' %args)
    
    tf.logging.set_verbosity('INFO')

    if args.hparams:
        hparams.parse(args.hparams)

    my_checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs=300,  # Save checkpoints every 10 minutes
        # save_checkpoints_steps=1000,
        # log_step_count_steps=10,
        keep_checkpoint_max=100,  # Retain the 10 most recent checkpoints.
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.model_dir,
        params=hparams,
        warm_start_from=args.warm,
        config=my_checkpointing_config
    )
    
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: tfrecord_input_fn(
        args.train, hparams, shuffle_buf=args.shuffle_buf, target=args.target), max_steps=args.train_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: tfrecord_input_fn(
        args.eval_, hparams, shuffle_buf=None, target=args.target), steps=100,throttle_secs=300)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RouteNet: a Graph Neural Network model for computer network modeling')

    subparsers = parser.add_subparsers(help='sub-command help')
    parser_data = subparsers.add_parser('data', help='data processing')
    parser_data.add_argument('-d', help='data file',
                             type=str, required=True, nargs='+')
    # parser_data.set_defaults(func=data)

    parser_train = subparsers.add_parser('train', help='Train options')
    parser_train.add_argument(
        '--hparams', type=str, help='Comma separated list of "name=value" pairs.')
    parser_train.add_argument(
        '--train', help='Train Tfrecords files', type=str, nargs='+')
    parser_train.add_argument(
        '--eval_', help='Evaluation Tfrecords files', type=str, nargs='+')
    parser_train.add_argument('--model_dir', help='Model directory', type=str)
    parser_train.add_argument(
        '--train_steps', help='Training steps', type=int, default=100)
    parser_train.add_argument(
        '--eval_steps', help='Evaluation steps, defaul None= all', type=int, default=None)
    parser_train.add_argument(
        '--shuffle_buf', help="Buffer size for samples shuffling", type=int, default=10000)
    parser_train.add_argument(
        '--target', help="Predicted variable", type=str, default='delay')
    parser_train.add_argument(
        '--warm', help="Warm start from", type=str, default=None)
    parser_train.set_defaults(func=train)
    parser_train.set_defaults(name="Train")
    
    args = parser.parse_args()
    # print('args = ', args)
    args.func(args)
