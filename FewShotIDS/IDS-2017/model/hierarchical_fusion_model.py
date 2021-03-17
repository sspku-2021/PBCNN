# -*- coding: utf-8 -*-

import sys
sys.path.append("../")


from model.base_model import BaseModel
from model.utils.modules import mlp, cnn_lstm

import tensorflow as tf


class HierarchicalFusionModel(BaseModel):
    def __init__(self, session_length, height, width,
                 num_labels, learning_rate,
                 filter_sizes, num_filters,
                 lstm_size, lstm_num_layers, lstm_dropout_rate, attention_size,
                 average = False,
                 is_pretrain=False, is_time=False, is_size=False,
                 uncertainty=False, is_train = True, is_length = True, early_stop = True, is_tuning = False):

        super().__init__(session_length, height, width,
                         num_labels, learning_rate,
                         is_train = is_train, is_pretrain = is_pretrain,
                         is_time = is_time, is_size = is_size,
                         uncertainty = uncertainty, is_length = is_length, early_stop = early_stop, is_tuning = is_tuning)

        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.lstm_size = lstm_size
        self.lstm_num_layer = lstm_num_layers
        self.lstm_dropout_rate = lstm_dropout_rate
        self.attention_size = attention_size
        self.average = average


    def network(self, inputs, lengths, attention_fn = None):
        outputs, state = cnn_lstm(inputs, self.filter_sizes, self.num_filters, lengths,
                                  self.lstm_size, self.lstm_num_layer, self.dropout_rate,
                                  self.lstm_dropout_rate)
        if attention_fn:
            feature_vec1 = attention_fn(outputs, state, size=self.attention_size, weight_decay=self.l2_regularizer)
        else:
            if self.average:
                feature_vec1 = tf.reduce_sum(outputs/self.length, axis=1)
            else:
                feature_vec1 = state
        feature_vec2 = mlp(feature_vec1, self.num_layers, self.hidden_size, self.dropout_rate, self.is_train,
                           weight_decay=self.l2_regularizer)
        return feature_vec1, feature_vec2

    def build_model(self, attention_fn = None):
        self.placeholder()

        with tf.variable_scope("embedding_module", reuse=tf.AUTO_REUSE) as scope:
            self.feature_vec, output = self.network(self.session, self.length, attention_fn)

        if self.is_pretrain:
            self.pretrain(output)

        else:
            self._cat_loss, self._category_predicts = self.category_loss(output, self.label, scope="category")
            self.cal_cost(self._cat_loss)
            #self.cal_acc(predicts=self._category_predicts)

        self.get_train_ops(self.cost, self.is_tuning)

    # def assign_lstm_kp(self, session, kp_value):
    #     session.run(tf.assign(self.lstm_dropout_rate, kp_value))

if __name__ == "__main__":
    from model.utils.modules import self_attention, self_attention_last
    model = HierarchicalFusionModel(24, 32, 32,
                                    7, 2e-4,
                                    [3, 4, 5, 6], num_filters=64,
                                    lstm_size = 128, lstm_num_layers=1, lstm_dropout_rate=0.9, attention_size=128)
    model.build_model(attention_fn=self_attention)




