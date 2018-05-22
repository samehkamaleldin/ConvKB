import tensorflow as tf
import numpy as np
import math


class ConvKB(object):

    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, sequence_length, num_classes, embedding_size, filter_sizes, num_filters, vocab_size,
                 pre_trained=[], l2_reg_lambda=0.001, batch_size=256, is_trainable=True, useConstantInit=False):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [batch_size, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.name_scope("embedding"):
            if pre_trained == []:
                self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -math.sqrt(1.0/embedding_size), math.sqrt(1.0/embedding_size), seed=1234), name="W")
            else:
                self.W = tf.get_variable(name="W2", initializer=pre_trained, trainable=is_trainable)

            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.tp_embed = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution
        pooled_outputs = []
        filter_size = 1

        with tf.name_scope("conv-%s" % filter_size):

            # filter constant initialiser
            pos = tf.ones([2, filter_size, 1, num_filters])
            neg = tf.ones([1, filter_size, 1, num_filters]) * -1
            weight_init = tf.concat([pos, neg], axis=0)
            W = tf.get_variable(name="W3", initializer=weight_init)
            b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name="b1")
            conv = tf.nn.conv2d(self.tp_embed, W, strides=[1, 1, 1, 1], padding="VALID", name="conv1")

            # Apply nonlinearity
            # Combine all the pooled features
            self.h_pool = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

        total_dims = num_filters * embedding_size
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, total_dims])

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):

            d_init = tf.contrib.layers.xavier_initializer(seed=1234)
            dense_init = tf.get_variable("W", shape=[total_dims, num_classes], initializer=d_init)
            dense_b = tf.Variable(tf.constant(0.0, shape=[num_classes]), name="b2")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.h_pool_flat, dense_init, dense_b, name="scores")
            self.predictions = tf.nn.sigmoid(self.scores)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softplus(self.scores * self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)
