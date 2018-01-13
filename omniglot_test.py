import argparse
import sys
import time
import tensorflow as tf
import numpy as np
from meta_batch_iterator import MetaBatchIterator
import datasets

from models import *
from utils import *


def convolutional_params(n_channels):
    W_conv = weight_variable([3, 3, n_channels, 64])
    b_conv = bias_variable([64])

    return W_conv, b_conv


def convolutional_block(W, b, x, training_ph, pool=False):

    no_pooled_out = tf.nn.relu(tf.layers.batch_normalization(conv2d(x, W) + b, training=training_ph))
    if pool:
        return max_pool_2x2(no_pooled_out)
    else:
        return tf.nn.relu(no_pooled_out)


# Basic model parameters as external flags.
FLAGS = None


def main(_):
    print(FLAGS)

    example_size = (28, 28, 1)

    batch_size = FLAGS.batch_size
    lr_init = FLAGS.learning_rate

    c_way = FLAGS.c_way
    k_shot = FLAGS.k_shot
    n_query = FLAGS.n_query_train
    n_train_steps = FLAGS.n_episodes_training // batch_size

    n_query_test = FLAGS.n_query_test
    n_test_episodes = FLAGS.n_episodes_test // batch_size
    test_interval = FLAGS.test_interval

    rotations = list(range(FLAGS.n_rotations))

    omniglot = datasets.Omniglot(root=FLAGS.omniglot_path, download=True, rotations=rotations,
                                 split=FLAGS.n_train_classes, example_size= example_size)

    train_batch_iterator = MetaBatchIterator(omniglot.train, batch_size, c_way, k_shot, n_query)
    test_batch_iterator = MetaBatchIterator(omniglot.test, batch_size, c_way, k_shot, n_query_test)

    with tf.Graph().as_default():

        x1, x2, y = train_batch_iterator.get_placeholders()  # shape = [batch_size, n_query*c_way, c_way, k_shot, ...]
        training_ph = tf.placeholder(tf.bool)

        # reshaping for training
        x1_t = tf.reshape(x1, [-1, *example_size])
        x2_t = tf.reshape(x2, [-1, *example_size])
        y_c = tf.reduce_mean(y, axis=3)
        y_t = tf.reshape(y_c, [-1, 1])

        # Embedding module
        n_channels = example_size[-1]

        W1, b1 = convolutional_params(n_channels)
        h11 = convolutional_block(W1, b1, x1_t, training_ph, pool=True)
        h12 = convolutional_block(W1, b1, x2_t, training_ph, pool=True)

        n_channels = 64

        W2, b2 = convolutional_params(n_channels)
        h21 = convolutional_block(W2, b2, h11, training_ph, pool=True)
        h22 = convolutional_block(W2, b2, h12, training_ph, pool=True)

        W3, b3 = convolutional_params(n_channels)
        h31 = convolutional_block(W3, b3, h21, training_ph)
        h32 = convolutional_block(W3, b3, h22, training_ph)

        W4, b4 = convolutional_params(n_channels)
        h41 = convolutional_block(W4, b4, h31, training_ph)
        h42 = convolutional_block(W4, b4, h32, training_ph)

        # sum embeddings of the same class (useful when k-shot > 1)
        h4_example_size = (example_size[0]//4, example_size[0]//4, n_channels)

        h41_r = tf.reshape(h41, [batch_size, -1, c_way, k_shot, *h4_example_size])
        h41_r = tf.reduce_sum(h41_r, axis=3)
        h41_r = tf.reshape(h41_r, [-1, *h4_example_size])

        h42_r = tf.reshape(h42, [batch_size, -1, c_way, k_shot, *h4_example_size])
        h42_r = tf.reduce_sum(h42_r, axis=3)
        h42_r = tf.reshape(h42_r, [-1, *h4_example_size])

        # depth concatenation
        h = tf.concat([h41_r, h42_r], axis=3)
        n_channels = n_channels*2

        # Relation Network
        W5, b5 = convolutional_params(n_channels)
        h5 = convolutional_block(W5, b5, h, training_ph, pool=True)

        n_channels = 64

        W6, b6 = convolutional_params(n_channels)
        h6 = convolutional_block(W6, b6, h5, training_ph, pool=True)

        # final dimension before fully connected layers (depends on the number and type of convolutions applied)
        fd = 1

        h6_example_size_flat = fd * fd * n_channels
        h6_flat = tf.reshape(h6, [-1, h6_example_size_flat])

        W_fc1 = weight_variable([h6_example_size_flat, 8])
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

        # Experiment dictionary that will be saved every test_interval steps
        omniglot_evaluation_dict = {'test_accuracy': [], 'train_accuracy': [], 'test_loss': [], 'train_loss': [],
                                    'step_time': [], 'flags': FLAGS}

        result_path = get_result_filepath(FLAGS.results_filename, subfolder='omniglot')

        with tf.Session() as sess:
            sess.run(init_op)
            start_time = time.time()

            for i in range(n_train_steps):

                inputs = train_batch_iterator.get_inputs()
                _, loss_value, acc_value = sess.run([train, loss, accuracy], feed_dict={x1: inputs[0],
                                                                                         x2: inputs[1],
                                                                                         y: inputs[2],
                                                                                        training_ph: True})

                if i % test_interval == 0:

                    accuracies = []
                    losses = []

                    print('(s: %d, e: %d) train: loss, acc : %.4f, %.4f' % (i, i * batch_size, loss_value, acc_value))

                    omniglot_evaluation_dict['train_loss'].append(loss_value)
                    omniglot_evaluation_dict['train_accuracy'].append(acc_value)

                    for j in range(n_test_episodes):
                        inputs = test_batch_iterator.get_inputs()
                        loss_value, acc_value = sess.run([loss, accuracy], feed_dict={x1: inputs[0],
                                                                                      x2: inputs[1],
                                                                                      y: inputs[2],
                                                                                      training_ph: False})
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

                    save_obj(result_path, omniglot_evaluation_dict)


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
        '--test_interval',
        type=int,
        default=500,
        help='number of steps before testing'
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
        '--n_rotations',
        type=int,
        default=4,
        help='number of rotations to consider to augment number of classes (min=1, max=4)'
    )

    parser.add_argument(
        '--n_train_classes',
        type=int,
        default=1200,
        help='number of classes for training (without considering rotations) (omniglot has 1623 classes)'
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

