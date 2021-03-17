# -*- coding: utf-8 -*-

import sys
sys.path.append("../")

import tensorflow as tf

from model.hierarchical_model import HierarchicalModel


class Siamese(HierarchicalModel):
    def __init__(self,  session_length, height, width,
                        learning_rate, filter_sizes, num_filters,
                        filter_sizes_hierarchical, num_fitlers_hierarchical, num_labels = None,
                        is_train = True, early_stop = True, is_tuning = True):

        super().__init__(session_length, height, width,
                         num_labels, learning_rate,
                         filter_sizes, num_filters,
                         filter_sizes_hierarchical, num_fitlers_hierarchical,
                         is_train=is_train, is_tuning = is_tuning, early_stop = early_stop)
        """
        因为数据增强的缘故，该模型很容过拟合
        """

        self.dropout_rate = 0.5
        self.l2_regularizer = 0.01
        self.last_hidden_size = 128


    def placeholder(self):
        self.x1 = tf.placeholder(tf.float32, shape=[None, self.session_length, self.height, self.width], name="session1")
        self.x2 = tf.placeholder(tf.float32, shape=[None, self.session_length, self.height, self.width], name="session2")
        self.l1 = tf.placeholder(tf.int32, shape = [None], name="length1")
        self.l2 = tf.placeholder(tf.int32, shape = [None], name="length2")
        self.label = tf.placeholder(tf.int32, shape=[None], name="label")

    def inference_placeholder(self, K):
        self.x1 = tf.placeholder(tf.float32, shape=[1, self.session_length, self.height, self.width], name="session1")
        self.x2 = tf.placeholder(tf.float32, shape=[K, self.session_length, self.height, self.width], name="session2")
        self.l1 = tf.placeholder(tf.int32, shape=[1], name="length1")
        self.l2 = tf.placeholder(tf.int32, shape = [K], name="length2")

    def inference_build_model(self, K):
        self.inference_build_model()
        with tf.variable_scope("embedding_module", reuse=tf.AUTO_REUSE) as scope:
            o1, _ = self.network(self.x1, self.l1)
            scope.reuse_variables()
            o2, _ = self.network(self.x2, self.l2)
        o1 = tf.tile(o1, [K, 1])
        o1 = tf.nn.dropout(o1, keep_prob=self.dropout_rate)
        o2 = tf.nn.dropout(o2, keep_prob=self.dropout_rate)
        inputs = tf.concat([o1, o2, tf.expand_dims(dist, axis=-1)], axis=-1)
        with tf.variable_scope("MLP"):
            fc = tf.layers.dense(inputs=inputs,
                                 units=self.last_hidden_size,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer),
                                 activation=tf.nn.relu)
            fc = tf.nn.dropout(fc, keep_prob=self.dropout_rate)
            self.logits = tf.layers.dense(inputs=fc,
                                     units=2,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer))

    def build_model(self):
        self.placeholder()
        with tf.variable_scope("embedding_module", reuse=tf.AUTO_REUSE) as scope:
            o1, _ = self.network(self.x1, self.l1)
            scope.reuse_variables()
            o2, _ = self.network(self.x2, self.l2)
        dist = tf.sqrt(tf.reduce_sum(tf.square(o1 - o2) + 1e-6, axis=-1))
        # self.loss = self.margin_loss(dist, self.label)
        # self.cal_cost(self.loss)
        # self.cal_acc(dist)
        # self.get_train_ops(self._cost)
        # self.inference = dist
        o1 = tf.nn.dropout(o1, keep_prob=self.dropout_rate)
        o2 = tf.nn.dropout(o2, keep_prob=self.dropout_rate)
        inputs = tf.concat([o1, o2, tf.expand_dims(dist, axis=-1)], axis=-1)
        with tf.variable_scope("MLP"):
            fc = tf.layers.dense(inputs=inputs,
                                 units=self.last_hidden_size,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer),
                                 activation=tf.nn.relu)
            fc = tf.nn.dropout(fc, keep_prob=self.dropout_rate)
            self.logits = tf.layers.dense(inputs=fc,
                                     units=2,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularizer))
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.label), name="loss")

        self.cal_cost(loss)
        with tf.variable_scope("acc"):
            correct = tf.nn.in_top_k(logits, self.label, 1)
            self._acc = tf.reduce_mean(tf.cast(correct, tf.float32))

        self.get_train_ops(self._cost)

    def cos_sim(self, v1, v2):
        norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
        norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
        dot_products = tf.reduce_sum(v1 * v2, axis=1, name='cos_sim')
        return dot_products / (norm1 * norm2)

    def margin_loss(self, dist, label, margin = 5):
        dist += 1e-6
        pos = label * dist
        neg = (1 - label) * tf.square(tf.maximum(0.0, margin-dist))
        return tf.reduce_mean(pos + neg)

    # def cal_acc(self, dists):
    #     preds = tf.cast(dists < 0.5, tf.float32)
    #     correct_prediction = tf.cast(tf.equal(self.label, preds), tf.float32)
    #     self._acc = tf.reduce_mean(correct_prediction)

    def gen_feed_dict(self, batcher):
        feed_dict = {
            self.x1: batcher.session1,
            self.x2: batcher.session2,
            self.l1: batcher.length1,
            self.l2: batcher.length2,
            self.label: batcher.label
        }
        return feed_dict

    @property
    def acc(self):
        return self._acc


if __name__ == "__main__":
    model = Siamese(
        24, 32, 32,
        1e-4, [3, 4, 5], 64,
        [4, 5, 6], 128,
    )
    model.build_model()


