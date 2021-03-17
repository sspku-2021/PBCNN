# -*- coding: utf-8 -*-

import sys
sys.path.append("../")


from model.base_model import BaseModel
from model.utils.modules import mlp, hierarchical_1d_CNN


import tensorflow as tf


class HierarchicalModel(BaseModel):

    def __init__(self, session_length, height, width,
                 num_labels, learning_rate,
                 filter_sizes, num_filters,
                 filter_sizes_hierarchical, num_fitlers_hierarchical,
                 is_train = True, is_pretrain=False,
                 is_time=False, is_size=False,
                 uncertainty = False, is_length = True, early_stop = True, is_tuning = False):
        super().__init__(session_length, height, width,
                         num_labels, learning_rate,
                         is_train = is_train, is_pretrain = is_pretrain,
                         is_time = is_time, is_size = is_size,
                         uncertainty = uncertainty, is_length = is_length, early_stop=early_stop,
                         is_tuning = is_tuning)
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.filter_sizes_hierarchical = filter_sizes_hierarchical
        self.num_fitlers_hierarchical = num_fitlers_hierarchical

    def network(self, inputs, length):
        feature_vec1 =  hierarchical_1d_CNN(inputs, length,
                                            self.filter_sizes, self.num_filters,
                                            self.filter_sizes_hierarchical, self.num_fitlers_hierarchical,
                                            self.dropout_rate)
        feature_vec2 = mlp(feature_vec1,  self.num_layers, self.hidden_size, self.dropout_rate, self.is_train,
                          weight_decay=self.l2_regularizer)
        return feature_vec1, feature_vec2

    def build_model(self):
        self.placeholder()

        with tf.variable_scope("embedding_module", reuse=tf.AUTO_REUSE) as scope:
            self.feature_vec, output = self.network(self.session, self.length)

        if self.is_pretrain:
          self.pretrain(output)

        else:
            self._cat_loss, self._category_predicts = self.category_loss(output, self.label, scope="category")
            self.cal_cost(self._cat_loss)

        self.get_train_ops(self._cost, self.is_tuning)


if __name__ == "__main__":
    model = HierarchicalModel(24, 32, 32,
                              7, 2e-5,
                             [4, 5, 6], 64,
                             [3, 4, 5], 128)
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        model.build_model()
        saver = tf.train.Saver(tf.global_variables())
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        pass




