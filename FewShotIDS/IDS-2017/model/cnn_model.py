# -*- coding: utf-8 -*-

import sys
sys.path.append("../")


from model.base_model import BaseModel
from model.utils.modules import mlp

import tensorflow as tf



class CnnModel(BaseModel):

    def __init__(self, session_length, height, width,
                 num_labels, learning_rate,
                 num_output, is_pretrain = False, is_time = False, is_size = False,
                 is_train = True, uncertainty = False, early_stop = True, is_tuning = False
                 ):

        super().__init__(session_length, height, width,
                         num_labels, learning_rate,
                         is_train = is_train, is_pretrain = is_pretrain,
                         is_time = is_time, is_size = is_size,
                         uncertainty = uncertainty, early_stop=early_stop, is_tuning = is_tuning)
        self.num_output = num_output

    def network(self, inputs, model_fn):
        feature_vec1 = model_fn(inputs, self.num_output, is_train=self.is_train, weight_decay=self.l2_regularizer)
        feature_vec2 = mlp(feature_vec1, self.num_layers, self.hidden_size, self.dropout_rate, self.is_train, weight_decay=self.l2_regularizer)
        return feature_vec1, feature_vec2

    def build_model(self, model_fn):
        """
        :param model_fn: vgg16 / resnet / mobilenet ()
        :return:
        """
        self.placeholder()

        with tf.variable_scope("embedding_module", reuse=tf.AUTO_REUSE) as scope:
            self.feature_vec, output = self.network(self.session, model_fn)

        if self.is_pretrain:
            self.pretrain(output)

        else:
            self._cat_loss, self._category_predicts = self.category_loss(output, self.label, scope="category")
            self.cal_cost(self._cat_loss)
            #self.cal_acc(predicts=self._category_predicts)

        self.get_train_ops(self._cost, self.is_tuning)


if __name__ == "__main__":
    from model.utils.modules import vgg, resnet, mobile_net

    model = CnnModel(session_length=24, height=32, width=32, num_labels=7, learning_rate=1e-7,\
                     num_output=64, is_pretrain=True, is_time=True, is_size=True,\
                     is_train=True, uncertainty=True)
    model.build_model(mobile_net)





