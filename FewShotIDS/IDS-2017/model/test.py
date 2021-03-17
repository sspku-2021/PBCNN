# -*- coding: utf-8 -*-

import sys
sys.path.append("../")

from model.utils.data_generator import DataGenerator
from model.base_model import BaseModel
import config
import utils

from typing import Dict
import tensorflow  as tf
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
import pickle



def test(model: BaseModel, label_mapping: Dict,
          batch_size: int,  test_file_path: str,
          model_name: str, restore_ckpt: str,
          file_out: str, **kwargs):

    print("starting testing %s model" % model_name)

    gpu_config = utils.gpu_config()
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        if kwargs.get("func"):
            model.build_model(kwargs["func"])
        else:
            model.build_model()
        saver = tf.train.Saver(tf.global_variables())



    with tf.Session(graph=graph, config=gpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, restore_ckpt)
        print("Restored from", restore_ckpt)

        model.is_train = False
        model.dropout_rate = 1
        if model.is_length:
            model.lstm_dropout_rate = 1

        generator = DataGenerator(path=test_file_path,
                                  batch_size=batch_size,
                                  is_label=not model.is_pretrain,
                                  is_time=model.is_time,
                                  is_size=model.is_size,
                                  is_length=model.is_length)
        batcher = generator.batch_generator(label_mapping)

        if model.is_pretrain:
            if model.is_time:
                time_predicts, time_true = [], []
            if model.is_size:
                size_predicts, size_true = [], []
        else:
            cat_predicts = []
            cat_true = []

        while True:
            next_batch = next(batcher)
            if next_batch is None:
                break
            feed_dict = model.gen_feed_dict(next_batch)
            if model.is_pretrain:
                if model.is_time and model.is_size:
                    predict_time, predict_size = sess.run([model.time_predict, model.size_predict], feed_dict=feed_dict)
                elif model.is_time:
                    predict_time = sess.run(model.time_predict, feed_dict=feed_dict)
                else:
                    predict_size = sess.run(model.size_predict, feed_dict=feed_dict)

                if model.is_time:
                    time_predicts.extend(predict_time)
                    time_true.extend(next_batch.duration_time)

                if model.is_size:
                    size_predicts.extend(predict_size)
                    size_true.extend(next_batch.session_size)
            else:
                predict_cat = sess.run(model.predict_category, feed_dict=feed_dict)
                cat_predicts.extend(predict_cat)
                cat_true.extend(next_batch.label)

        f_out = open(file_out, "w", encoding="utf-8")

        if model.is_pretrain:
            if model.is_time:
                print("time mse: {}".format(mean_squared_error(y_true=time_true, y_pred=time_predicts)))
            if model.is_size:
                print("size mse: {}".format(mean_squared_error(y_true=size_true, y_pred=size_predicts)))

            if model.is_time:
                for time_p, time_t in zip(time_predicts, time_true):
                    f_out.write(str(time_p) + "\t" + str(time_t) + "\n")
            if model.is_size:
                for size_p, size_t in zip(size_predicts, size_true):
                    f_out.write(str(size_p) + "\t" + str(size_t) + "\n")

        else:
            label_mapping_reverse = {v:k for k, v in label_mapping.items()}
            target_names = [label_mapping_reverse[i] for i in range(len(label_mapping_reverse))]
            print(classification_report(cat_true, cat_predicts, target_names = target_names, digits=4))
            for pred, label in zip(cat_predicts, cat_true):
                pred = label_mapping_reverse[int(pred)]
                label = label_mapping_reverse[int(label)]
                f_out.write(pred + "\t" + label + "\n")

        f_out.close()






