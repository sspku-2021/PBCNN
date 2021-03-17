# -*- coding: utf-8 -*-

import sys
sys.path.append("../")

import tensorflow as tf
import os
import math
from typing import Dict
import logging

import config
import utils
from model.base_model import BaseModel
from model.utils.keras_generic_utils import Progbar
from model.utils.data_generator import DataGenerator, input_fn, DataGeneratorPair
from model.utils.utils import get_model_params, restore_model_params


def run_epoch_siamese(sess, model: BaseModel, batcher, epoch: int = -1):
    step = 0
    losses, iters = 0., 0.
    acces = 0.

    while True:
        next_batch = next(batcher)
        if next_batch is None:
            break
        feed_dict = model.gen_feed_dict(next_batch)
        if model.is_train:
            loss, acc, step, _ = sess.run([model.cost_non_reg, model.acc, model.global_step, model.train_ops], feed_dict=feed_dict)
        else:
            loss, acc  = sess.run([model.cost_non_reg, model.acc], feed_dict=feed_dict)

        losses += loss
        acces += acc
        iters += 1

        if step % 1 == 0 and model.is_train:
            logging.info("Epoch %d, Batch %d, Global step %d:" % (epoch, iters, step))
            logging.info("training batch loss: %.4f; avg_loss: %.4f" % (loss, losses / iters))
            logging.info("training batch acc: %.4f; avg_acc: %.4f" % (acc, acces / iters))
            logging.info("")



    return loss, losses / iters, acc, acces / iters, step


def run_epoch(sess, model: BaseModel, batcher, epoch: int = -1):
    step = 0
    losses, iters = 0., 0.
    if not model.is_pretrain:
        acces = 0.

    while True:
        next_batch = next(batcher)
        if next_batch is None:
            break
        feed_dict = model.gen_feed_dict(next_batch)
        if model.is_pretrain:
            if model.is_train:
                loss, step, _ = sess.run([model.cost_non_reg, model.global_step, model.train_ops], feed_dict=feed_dict)
            else:
                loss = sess.run(model.cost_non_reg, feed_dict=feed_dict)

        else:
            if model.is_train:
                loss, predict, step, _ = sess.run([model.cost_non_reg, model.predict_category, model.global_step, model.train_ops], feed_dict=feed_dict)
            else:
                loss, predict = sess.run([model.cost_non_reg, model.predict_category], feed_dict=feed_dict)
            acc = len(next_batch.label[(next_batch.label == predict)]) / len(next_batch.label)
        losses += loss
        iters += 1
        if step % 100 == 0 and model.is_train:
            logging.info("Epoch %d, Batch %d, Global step %d:" % (epoch, iters, step))
            logging.info("training batch loss: %.4f; avg_loss: %.4f" % (loss, losses / iters))
            if not model.is_pretrain:
                logging.info("training batch accuracy: %.4f; avg_accuracy: %.4f" % (acc, acces / iters))
            logging.info("")
        if not model.is_pretrain:
            acces += acc
        # if model.is_train and iters < num_of_batches - 1:
        #     if model.is_pretrain:
        #         probar.update((int(iters)) * batch_size, [('loss', loss), ("avg_loss", losses / iters)])
        #     else:
        #         probar.update((int(iters)) * batch_size, [('loss', loss), ("avg_loss", losses / iters), ("acc", acc), ("avg_acc", acces/iters)])

    if model.is_pretrain:
        return loss, losses / iters, step
    else:
        return loss, losses / iters, acc, acces / iters, step


def train_siamese(model, batch_size, max_epoch,
                  train_file_path, dev_file_path,
                  is_tuning,
                  ckpt_dir, restore_ckpt = None,
                  start_epoch = -1):
    print("starting training siamese model.")
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    chechpoint_prefix = os.path.join(ckpt_dir, "siamese")
    gpu_config = utils.gpu_config()
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        model.build_model()
        if is_tuning:
            variables = [var for var in tf.trainable_variables() if var.name.startswith("embedding_module")]
            saver = tf.train.Saver(var_list=variables)
        else:
            saver = tf.train.Saver(tf.global_variables())

    with tf.Session(graph=graph, config=gpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        if restore_ckpt:
            saver.restore(sess, restore_ckpt)
            print("Restored from", restore_ckpt)
        elif start_epoch != -1:
            restore_ckpt = os.path.join(ckpt_dir, "siamese_" + str(start_epoch))
            saver.restore(sess, restore_ckpt)
            print("Restored from", restore_ckpt)

        dropout_restore = model.dropout_rate
        eval_losses = []
        best_model_params = None
        for epoch in range(start_epoch+1, max_epoch):
            restore_ckpt = os.path.join(ckpt_dir, "siamese_" + str(epoch))
            model.dropout_rate = dropout_restore
            model.is_train = True
            generator = DataGeneratorPair(path = train_file_path,
                                          batch_size = batch_size)
            batcher = generator.batch_generator()
            loss, avg_loss, acc, avg_acc, global_steps = run_epoch_siamese(sess, model, batcher, epoch)
            logging.info("Epoch %d, training batch loss: %.4f; avg_loss: %.4f" % (epoch, loss, avg_loss))
            logging.info("Epoch %d, training batch accuracy: %.4f; avg_accuracy: %.4f" % (epoch, acc, avg_acc))
            logging.info("")

            model.is_train = False
            model.dropout_rate = 1

            dev_generator = DataGeneratorPair(path = dev_file_path,
                                              batch_size = batch_size)
            dev_batcher = dev_generator.batch_generator()
            dev_loss, dev_avg_loss, dev_acc, dev_avg_acc, _ = run_epoch_siamese(sess, model, dev_batcher)

            logging.info("Epoch %d, evaluating batch loss: %.4f; avg_loss: %.4f" % (epoch, dev_loss, dev_avg_loss))
            logging.info("Epoch %d, evaluating batch accuracy: %.4f; avg_accuracy: %.4f" % (epoch, dev_acc, dev_avg_acc))
            logging.info("")
            eval_losses.append(dev_avg_loss)
            if model.early_stop:
                if best_model_params == None:
                    best_model_params = get_model_params()
                elif dev_avg_loss == max(eval_losses):
                    best_model_params = get_model_params()
                if _early_stop(eval_losses, epoch):
                    restore_model_params(best_model_params)
                    path = saver.save(sess, chechpoint_prefix, global_step = global_steps)
                    print("Saved model checkpoint to {}\n".format(path))
                    return

            saver.save(sess, restore_ckpt)

def train(model: BaseModel, label_mapping: Dict,
          batch_size: int, max_epoch: int,
          train_file_path: str, dev_file_path: str,
          model_name: str, is_tuning: int = False,
          ckpt_dir: str = config.ckpt_dir, restore_ckpt: str = None,
          start_epoch = -1, **kwargs):
    print("starting training %s model" % model_name)
    ckpt_path = os.path.join(ckpt_dir, model_name)
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    chechpoint_prefix = os.path.join(ckpt_path, model_name)

    gpu_config = utils.gpu_config()
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        if kwargs.get("func"):
            model.build_model(kwargs["func"])
        else:
            model.build_model()
        if is_tuning:
            variables = [var for var in tf.trainable_variables() if var.name.startswith("embedding_module")]
            saver = tf.train.Saver(var_list=variables)
        else:
            saver = tf.train.Saver(tf.global_variables())

    with tf.Session(graph=graph, config=gpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        if restore_ckpt:
            saver.restore(sess, restore_ckpt)
            print("Restored from", restore_ckpt)
        elif start_epoch != -1:
            restore_ckpt = os.path.join(ckpt_path, model_name + "_" + str(start_epoch))
            saver.restore(sess, restore_ckpt)
            print("Restored from", restore_ckpt)

        dropout_restore = model.dropout_rate
        if hasattr(model, "lstm_dropout_rate"):
            lstm_dropout_restore = model.lstm_dropout_rate

        eval_losses = []
        best_model_params = None
        for epoch in range(start_epoch+1, max_epoch):
            restore_ckpt = os.path.join(ckpt_path, model_name + '_' + str(epoch))
            model.dropout_rate = dropout_restore
            model.is_train = True
            if hasattr(model, "lstm_dropout_rate"):
                model.lstm_dropout_rate = lstm_dropout_restore

            # batcher = input_fn(train_file_path, label_mapping, batch_size,
            #                      num_epochs = 1, is_label = not model.is_pretrain, is_length = model.is_length,
            #                      is_size = model.is_size, is_time = model.is_time)

            generator = DataGenerator(path = train_file_path,
                                      batch_size = batch_size,
                                      is_label = not model.is_pretrain,
                                      is_time = model.is_time,
                                      is_size = model.is_size,
                                      is_length = model.is_length)
            batcher = generator.batch_generator(label_mapping)
            if model.is_pretrain:
                loss, avg_loss, global_steps = run_epoch(sess, model, batcher, epoch)
            else:
                loss, avg_loss, acc, avg_acc, global_steps = run_epoch(sess, model, batcher, epoch)

            if not model.is_pretrain:
                logging.info("training batch accuracy: %.4f; avg_accuracy: %.4f" % (acc, avg_acc))
                logging.info("")

            model.is_train = False
            model.dropout_rate = 1
            if hasattr(model, "lstm_dropout_rate"):
                model.lstm_dropout_rate = 1

            dev_generator = DataGenerator(path = dev_file_path,
                                          batch_size = batch_size,
                                          is_label = not model.is_pretrain,
                                          is_time = model.is_time,
                                          is_size = model.is_size,
                                          is_length = model.is_length)
            dev_batcher = dev_generator.batch_generator(label_mapping)
            if model.is_pretrain:
                dev_loss, dev_avg_loss, _ = run_epoch(sess, model, dev_batcher)
            else:
                dev_loss, dev_avg_loss, dev_acc, dev_avg_acc, _ = run_epoch(sess, model, dev_batcher)

            logging.info("Epoch %d, evaluating batch loss: %.4f; avg_loss: %.4f" % (epoch, dev_loss, dev_avg_loss))
            if not model.is_pretrain:
                logging.info("evaluating batch accuracy: %.4f; avg_accuracy: %.4f" % (dev_acc, dev_avg_acc) + "\n")
            logging.info("")
            # log_values = []
            # log_values.append(("loss", loss))
            # log_values.append(("avg_loss", avg_loss))
            # log_values.append(("val_loss", dev_avg_loss))
            # probar.update(generator.cnt, log_values)
            eval_losses.append(dev_avg_loss)
            if model.early_stop:
                if best_model_params == None:
                    best_model_params = get_model_params()
                elif dev_avg_loss == max(eval_losses):
                    best_model_params = get_model_params()
                if _early_stop(eval_losses, epoch):
                    restore_model_params(best_model_params)
                    path = saver.save(sess, chechpoint_prefix, global_step = global_steps)
                    print("Saved model checkpoint to {}\n".format(path))
                    return

            saver.save(sess, restore_ckpt)

def _early_stop(eval_losses, cur_epoch, min_epoch=2, stop_after=3):
    if cur_epoch < min_epoch:
        return False
    n = len(eval_losses)
    if n < stop_after:
        return False
    for idx in range(n - stop_after + 1, n):
        if eval_losses[idx] < eval_losses[idx - 1]:
            return False
    print("eval_loss in count: ", eval_losses)
    return True

