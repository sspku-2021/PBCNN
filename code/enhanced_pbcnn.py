import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Embedding, TimeDistributed
from tensorflow.keras import Input
from tensorflow.keras import layers, Model
from tensorflow import keras as K

import pickle
import shutil
import time
import matplotlib.pyplot as plt
from absl import logging, app
from sklearn.metrics import classification_report

np.set_printoptions(threshold=np.inf)
AUTOTUNE = tf.data.experimental.AUTOTUNE

MAX_PKT_BYTES = 50 * 50
MAX_PKT_NUM = 100

pkt_num = 20
pkt_bytes = 256
num_class = 15

class MyLogCallback(K.callbacks.Callback):
    def __init__(self, valid_ds=None):
        super().__init__()
        #self._filename = file_name
        #self.logs_dic = {}
        self.validation_data = valid_ds
        #self.valid_freq = valid_req
        #self.batch_cnt = 0

    # def on_batch_end(self, batch, logs=None):
    #     self.batch_cnt += 1
    #     for k, v in logs.items():
    #         self.logs_dic.setdefault(k, []).append(v)
    #     if self.validation_data \
    #             and self.valid_freq > 0 \
    #             and batch % self.valid_freq == 0:
    #         accuracy, precision, recall, f1_score, val_loss = self._eval_valid(logs)
    #         self.logs_dic.setdefault('val_accuracy', []).append(accuracy)
    #         self.logs_dic.setdefault('val_f1_score', []).append(f1_score)
    #         self.logs_dic.setdefault('val_loss', []).append(val_loss)

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data:
            accuracy, precision, recall, f1_score, val_loss = self._eval_valid(logs)

            # self.logs_dic.setdefault('val_accuracy', []).append(accuracy)
            # self.logs_dic.setdefault('val_precision', []).append(precision)
            # self.logs_dic.setdefault('val_recall', []).append(recall)
            # self.logs_dic.setdefault('val_f1_score', []).append(f1_score)
            # self.logs_dic.setdefault('val_loss', []).append(val_loss)

    # def on_train_end(self, logs=None):
    #     with open(self._filename, 'wb') as fww:
    #         pickle.dump(self.logs_dic, fww)

    def _eval_valid(self, logs, digits=6):
        y_pred, y_true, losses = [], [], []
        cnt = 0
        for features, labels in self.validation_data:
            y_ = self.model.predict(features)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, y_)
            y_ = np.argmax(y_, axis=-1)
            y_pred.append(y_)
            y_true.append(labels.numpy())
            losses.append(loss.numpy().sum())
            cnt += len(labels)

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        cl_re = classification_report(y_true, y_pred, digits=digits, output_dict=True)
        accuracy = round(cl_re['accuracy'], digits)
        precision = round(cl_re['macro avg']['precision'], digits)
        recall = round(cl_re['macro avg']['recall'], digits)
        f1_score = round(cl_re['macro avg']['f1-score'], digits)

        logs['val/accuracy'] = accuracy
        logs['val/precision'] = precision
        logs['val/recall'] = recall
        logs['val/f1_score'] = f1_score
        logs['val/loss'] = sum(losses) / cnt
        #print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (f1_score, precision, recall))

        return accuracy, precision, recall, f1_score, sum(losses) / cnt

def parse_sparse_example(example_proto):
    features = {
        'sparse': tf.io.SparseFeature(index_key=['idx1', 'idx2'],
                                      value_key='val',
                                      dtype=tf.int64,
                                      size=[MAX_PKT_NUM, MAX_PKT_BYTES]),
        'label': tf.io.FixedLenFeature([], dtype=tf.int64),
        'byte_len': tf.io.FixedLenFeature([], dtype=tf.int64),
        'last_time': tf.io.FixedLenFeature([], dtype=tf.float32),
    }
    batch_sample = tf.io.parse_example(example_proto, features)
    sparse_features = batch_sample['sparse']
    labels = batch_sample['label']
    sparse_features = tf.sparse.slice(sparse_features, start=[0, 0], size=[pkt_num, pkt_bytes])
    dense_features = tf.sparse.to_dense(sparse_features)
    #dense_features = tf.cast(dense_features, tf.float32) / 255.
    return dense_features, labels

def generate_ds(path_dir, batch_size, use_cache=False):
    assert os.path.isdir(path_dir)
    ds = tf.data.Dataset.list_files(os.path.join(path_dir, '*.tfrecord'), shuffle=True)
    ds = ds.interleave(
        lambda x: tf.data.TFRecordDataset(x).map(parse_sparse_example),
        cycle_length=AUTOTUNE,
        block_length=8,
        num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    if use_cache:
        ds = ds.cache()
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def encoder(embedded_packet):
    x = layers.Conv1D(filters=8, kernel_size=3, strides=1, padding='same', data_format=data_format)(embedded_packet)
    x = layers.BatchNormalization(axis=-1 if data_format == 'channels_last' else 1)(x)
    x = layers.Activation(activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same', data_format='channels_last')(x)

    x = layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='same', data_format=data_format)(x)
    x = layers.BatchNormalization(axis=-1 if data_format == 'channels_last' else 1)(x)
    x = layers.Activation(activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same', data_format='channels_last')(x)

    x = layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='same', data_format=data_format)(x)
    x = layers.BatchNormalization(axis=-1 if data_format == 'channels_last' else 1)(x)
    x = layers.Activation(activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=8, strides=8, padding='same', data_format='channels_last')(x)

    y = layers.Flatten()(x)
    # y = layers.Dense(512, activation='relu')(y)
    # y = layers.Dense(256, activation='relu')(y)

    return y

def text_cnn_block(x, filters, height, width, data_format='channels_last'):
    x = layers.Conv2D(filters=filters, kernel_size=(height, width),
                      strides=1, data_format=data_format)(x)
    x = layers.BatchNormalization(axis=-1 if data_format == 'channels_last' else 1)(x)
    x = layers.Activation(activation='relu')(x)
    x = tf.reduce_max(x, axis=1, keepdims=False)
    return x
    
def pbcnn(encodedSession):
    y = tf.reshape(encodedSession, shape=(-1, pkt_num, pkt_bytes, 1))
    data_format = 'channels_last'
    y1 = text_cnn_block(y, filters=256, height=3, width=pkt_bytes)
    y2 = text_cnn_block(y, filters=256, height=4, width=pkt_bytes)
    y3 = text_cnn_block(y, filters=256, height=5, width=pkt_bytes)
    y = layers.concatenate(inputs=[y1, y2, y3], axis=-1)

    y = layers.Flatten(data_format=data_format)(y)
    y = layers.Dense(512, activation='relu')(y)
    y = layers.Dense(256, activation='relu')(y)
    # y = layers.Dense(128, activation='relu')(y)
    y = layers.Dense(num_class, activation='linear')(y)

    return y

def eval_ds(self, model_dir, test_ds=None, digits=6):
    print('predict...')
    checkpoint_save_path = model_dir
    if os.path.exists(checkpoint_save_path):
        print('----------------load the model-------------------')
        # model.load_weights(checkpoint_save_path)
        model = tf.keras.models.load_model(checkpoint_save_path)
    print('model {} has been loaded.'.format(model_dir))
    
    y_pred, y_true = [], []
    for features, labels in test_ds:
        y_ = model.predict(features) # xuezhang
        #y_ = model(features) # added by lizhao 2020/10/14
        y_ = np.argmax(y_, axis=-1)
        y_pred.append(y_)
        y_true.append(labels.numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    label_names = ['ftp-bruteforce', 'ddos-hoic', 'dos-goldeneye', 'ddos-loic-http', 'sql-injection',
                   'dos-hulk', 'bot', 'ssh-bruteforce', 'bruteforce-xss', 'dos-slowhttptest',
                   'bruteforce-web', 'dos-slowloris', 'benign', 'ddos-loic-udp', 'infiltration']

    cl_re = classification_report(y_true, y_pred, digits=digits,
                                  labels=[i for i in range(self._num_class)],
                                  target_names=label_names, output_dict=True)
    accuracy = round(cl_re['accuracy'], digits)
    precision = round(cl_re['macro avg']['precision'], digits)
    recall = round(cl_re['macro avg']['recall'], digits)
    f1_score = round(cl_re['macro avg']['f1-score'], digits)

    # print(f'Macro Avg')
    # print(f'Accuracy: \t{accuracy} \n'
    #       f'Precision: \t{precision} \n'
    #       f'Recall: \t{recall} \n'
    #       f'F1-Score: \t{f1_score}')
    # # plot_heatmap(cl_re, y_labels=label_names)
    # return y_true, y_pred, cl_re

    return accuracy, precision, recall, f1_score, cl_re

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

    train_path = "/home/fgtc/712_tfrecord/train"
    valid_path = "/home/fgtc/712_tfrecord/valid"
    test_path = "/home/fgtc/712_tfrecord/test"
    batch_size = 512
    data_format = 'channels_last'

    train_ds = generate_ds(train_path, batch_size)
    valid_ds = generate_ds(valid_path, batch_size)
    test_ds = generate_ds(test_path, batch_size)

    embedding_layer = Embedding(256, 256, input_length=256, trainable=True, mask_zero=True)

    embedded_packet = Input(shape=(256, 256), dtype='float32') # (256, 256)
    y = encoder(embedded_packet)
    packetEncoder = Model(embedded_packet, y)
    packetEncoder.summary()

    session_input = Input(shape=(20, 256), dtype='int32')
    #embedded_session = embedding_layer(session_input) # output shape: (20, 256, 256)
    embedded_session = tf.one_hot(session_input, 256, on_value=1.0, off_value=0.0, axis=-1) # output shape: (20, 256, 256)
    encodedSession = TimeDistributed(packetEncoder)(embedded_session) # output shape: (20, 256)
    pred = pbcnn(encodedSession)
    model = Model(session_input, pred)
    model.summary()

    model.compile(optimizer=K.optimizers.Adam(learning_rate=0.001),
                            loss=K.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=[K.metrics.SparseCategoricalAccuracy(name='mean/acc'),
                                     K.metrics.SparseCategoricalCrossentropy(name='mean/loss', from_logits=True)])
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['sparse_categorical_accuracy'])

    checkpoint_save_path = "./checkpoint_enhanced_pbcnn/"
    if os.path.exists(checkpoint_save_path):
        print('----------------load the model-------------------')
        # model.load_weights(checkpoint_save_path)
        model = tf.keras.models.load_model(checkpoint_save_path)

    callbacks = []

    callbacks.append(MyLogCallback(batch_log, valid_ds))

    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                        save_weights_only=False,
                                                        save_best_only=True,
                                                        monitor='val/f1_score'))
    s = time.time()
    model.fit(train_ds,
              epochs=10,
              validation_data=valid_ds,
              callbacks=callbacks)
    training_10_epochs_time = (time.time() - s) / 60
    print('train 10 epochs time:', training_10_epochs_time, 'min')

    s1 = time.time()
    accuracy, precision, recall, f1_score, cl_re = eval_ds(model_dir=checkpoint_save_path, 
                                                           test_ds=test_ds)
    test_time = (time.time() - s1) / 60
    print('test time:', test_time, 'min')
    print('accuracy:', accuracy)
    print('precision:', precision)
    print('recall:', recall)
    print('f1_score:', f1_score)
    print('cl_re:', cl_re)