import argparse
import sys
import time
import tensorflow as tf
import numpy as np
from meta_batch_builder import MetaBatchBuilder
import datasets

from models import *
from utils import *


def convolutional_params(n_channels):
    W_conv = weight_variable([3, 3, n_channels, 64])
    b_conv = bias_variable([64])

    return W_conv, b_conv


def convolutional_block(W, b, x, pool=False):
    # TODO: add batch normalization

    if pool:
        return max_pool_2x2(tf.nn.relu(tf.contrib.layers.batch_norm(conv2d(x, W) + b)))
    else:
        return tf.nn.relu(conv2d(x, W) + b)


# Basic model parameters as external flags.
FLAGS = None


def main(_):
    print(FLAGS)

    input_size = 28
    n_channels = 1

    batch_size = FLAGS.batch_size
    lr_init = FLAGS.learning_rate

    c_way = FLAGS.c_way
    k_shot = FLAGS.k_shot
    n_query = FLAGS.n_query_train
    n_train_steps = FLAGS.n_episodes_training // batch_size

    n_query_test = FLAGS.n_query_test
    n_test_episodes = FLAGS.n_episodes_test // batch_size

    omniglot = datasets.Omniglot(root=FLAGS.omniglot_path, download=True)
    train_batch_builder = MetaBatchBuilder(omniglot.train, batch_size, c_way, k_shot, n_query)
    train_batch_builder.resize = input_size

    test_batch_builder = MetaBatchBuilder(omniglot.test, batch_size, c_way, k_shot, n_query_test)
    test_batch_builder.resize = input_size

    with tf.Graph().as_default():
        input_size = (train_batch_builder.resize, train_batch_builder.resize, n_channels)

        x1, x2, y = train_batch_builder.get_placeholders(input_size)


        # reshaping for training
        x1_t = tf.reshape(x1, [-1, *input_size])
        x2_t = tf.reshape(x2, [-1, *input_size])
        y_c = tf.reduce_mean(y, axis=3)
        y_t = tf.reshape(y_c, [-1, 1])

        # Embedding module

        W1, b1 = convolutional_params(n_channels)
        h11 = convolutional_block(W1, b1, x1_t, pool=True)
        h12 = convolutional_block(W1, b1, x2_t, pool=True)

        n_channels = 64

        W2, b2 = convolutional_params(n_channels)
        h21 = convolutional_block(W2, b2, h11, pool=True)
        h22 = convolutional_block(W2, b2, h12, pool=True)

        W3, b3 = convolutional_params(n_channels)
        h31 = convolutional_block(W3, b3, h21)
        h32 = convolutional_block(W3, b3, h22)

        W4, b4 = convolutional_params(n_channels)
        h41 = convolutional_block(W4, b4, h31)
        h42 = convolutional_block(W4, b4, h32)

        # sum embeddings of the same class
        h41_r = tf.reshape(h41, [batch_size, -1, c_way, k_shot, 7, 7, n_channels])
        h41_r = tf.reduce_sum(h41_r, axis=3)
        h41_r = tf.reshape(h41_r, [-1, 7, 7, n_channels])

        h42_r = tf.reshape(h42, [batch_size, -1, c_way, k_shot, 7, 7, n_channels])
        h42_r = tf.reduce_sum(h42_r, axis=3)
        h42_r = tf.reshape(h42_r, [-1, 7, 7, n_channels])

        h = tf.concat([h41_r, h42_r], axis=3)  # depth concatenation

        n_channels = n_channels*2

        # Relation Network
        W5, b5 = convolutional_params(n_channels)
        h5 = convolutional_block(W5, b5, h, pool=True)

        n_channels = n_channels//2

        W6, b6 = convolutional_params(n_channels)
        h6 = convolutional_block(W6, b6, h5, pool=True)

        fd = 4

        h6_flat = tf.reshape(h6, [-1, 64 * fd])

        W_fc1 = weight_variable([fd * 64, 8])
        b_fc1 = bias_variable([8])
        h_fc1 = tf.nn.relu(tf.nn.xw_plus_b(h6_flat, W_fc1, b_fc1))

        W_fc2 = weight_variable([8, 1])
        b_fc2 = bias_variable([1])
        output = tf.nn.xw_plus_b(h_fc1, W_fc2, b_fc2)

        output_normalized = tf.nn.sigmoid(output)

        # MSE loss
        loss = tf.reduce_mean(tf.squared_difference(output_normalized, y_t))

        # TEST tensors
        y_test = tf.reshape(y_c, [batch_size, -1, c_way])
        y_idx = tf.argmax(y_test, axis=2)

        output_test = tf.reshape(output, [batch_size, -1, c_way])
        max_idx = tf.argmax(output_test, axis=2)

        correct_prediction = tf.equal(y_idx, max_idx)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        # Optimizer
        opt = tf.train.AdamOptimizer(lr_init)
        train = opt.minimize(loss)

        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        omniglot_evaluation_dict = {'test_accuracy': [], 'train_accuracy': [], 'test_loss': [], 'train_loss': [],
                                    'step_time': [], 'flags': FLAGS}

        with tf.Session() as sess:
            sess.run(init_op)
            start_time = time.time()

            for i in range(n_train_steps):

                inputs = train_batch_builder.get_inputs()
                _, loss_value, acc_value = sess.run([train, loss, accuracy], feed_dict={x1: inputs[0],
                                                                                         x2: inputs[1],
                                                                                         y: inputs[2]})

                if i % 200 == 0:

                    accuracies = []
                    losses = []

                    print('(s: %d, e: %d) train: loss, acc : %.4f, %.4f' % (i, i * batch_size, loss_value, acc_value))

                    omniglot_evaluation_dict['train_loss'].append(loss_value)
                    omniglot_evaluation_dict['train_accuracy'].append(acc_value)

                    for j in range(n_test_episodes):
                        inputs = test_batch_builder.get_inputs()
                        loss_value, acc_value = sess.run([loss, accuracy], feed_dict={x1: inputs[0],
                                                                                      x2: inputs[1],
                                                                                      y: inputs[2]})
                        accuracies.append(acc_value)
                        losses.append(loss_value)

                    mean_acc = np.mean(accuracies)
                    std_acc = np.std(accuracies)
                    mean_loss = np.mean(losses)
                    std_loss = np.std(losses)

                    duration = time.time() - start_time
                    print('--test(%.2fs): loss, acc (%d es): %.4f, %.4f(%.2f)' % (duration, n_test_episodes, mean_loss,
                                                                            mean_acc, std_acc))

                    omniglot_evaluation_dict['test_accuracy'].append((mean_acc, std_acc))
                    omniglot_evaluation_dict['test_loss'].append((mean_loss, std_loss))
                    omniglot_evaluation_dict['step_time'].append((i, duration))

                    res_path = get_result_filepath(FLAGS.results_filename, subfolder='omniglot')
                    save_obj(res_path, omniglot_evaluation_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
        help='Initial learning rate.'
    )

    parser.add_argument(
        '--c_way',
        type=int,
        default=5,
        help='number of classes per episode'
    )

    parser.add_argument(
        '--k_shot',
        type=int,
        default=1,
        help='number of support example per class'
    )

    parser.add_argument(
        '--n_query_train',
        type=int,
        default=19,
        help='number of test example during a training episode'
    )

    parser.add_argument(
        '--n_query_test',
        type=int,
        default=1,
        help='number of test example during a test episode'
    )

    parser.add_argument(
        '--n_episodes_training',
        type=int,
        default=400000,
        help='Number of episode for training'
    )

    parser.add_argument(
        '--n_episodes_test',
        type=int,
        default=1000,
        help='Number of episode for testing'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size.'
    )
    parser.add_argument(
        '--omniglot_path',
        type=str,
        default='omniglot',
        help='Directory containing the omniglot dataset.'
    )

    parser.add_argument(
        '--results_filename',
        type=str,
        default='omniglot_evaluation.pickle',
        help='Filename to the results file'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

