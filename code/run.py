import os
import pickle
import shutil
import time
# import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import tensorflow as tf
from absl import logging, app
from sklearn.metrics import classification_report
from tensorflow import keras as K
from tensorflow.keras import Input
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Layer

MAX_PKT_BYTES = 50 * 50
MAX_PKT_NUM = 100
AUTOTUNE = tf.data.experimental.AUTOTUNE


def plot_heatmap(report, y_labels=None):
    mt = []
    if y_labels is None:
        y_labels = ['ftp-bruteforce', 'ddos-hoic', 'dos-goldeneye', 'ddos-loic-http', 'sql-injection',
                    'dos-hulk', 'bot', 'ssh-bruteforce', 'bruteforce-xss', 'dos-slowhttptest',
                    'bruteforce-web', 'dos-slowloris', 'benign', 'ddos-loic-udp', 'infiltration']
    support = []
    x_labels = ['precision', 'recall', 'f1-score']
    for name in y_labels:
        mt.append([
            report[name]['precision'],
            report[name]['recall'],
            report[name]['f1-score']
        ])
        support.append(report[name]['support'])
    assert len(support) == len(y_labels)
    y_labels_ = []
    for i in range(len(y_labels)):
        y_labels_.append(f'{y_labels[i]} ({support[i]})')
    plt.figure(figsize=(5, 6), dpi=200)
    sns.set()
    sns.heatmap(mt, annot=True, xticklabels=x_labels, yticklabels=y_labels_, fmt='.4f',
                linewidths=0.5, cmap='PuBu', robust=True)
    plt.show()


class TF(object):

    def __init__(self, pkt_bytes, pkt_num, model,
                 train_path, valid_path, test_path,
                 batch_size=128, num_class=15):
        model = model.lower().strip()
        assert pkt_bytes <= MAX_PKT_BYTES, f'Check pkt bytes less than max pkt bytes {MAX_PKT_BYTES}'
        assert pkt_num <= MAX_PKT_NUM, f'Check pkt num less than max pkt num {MAX_PKT_NUM}'
        assert model in ('pbcnn', 'en_pbcnn'), f'Check model type'

        self._pkt_bytes = pkt_bytes
        self._pkt_num = pkt_num
        self._model_type = model

        assert os.path.isdir(train_path)
        assert os.path.isdir(valid_path)
        assert os.path.isdir(valid_path)

        self._train_path = train_path
        self._valid_path = valid_path
        self._test_path = test_path

        self._batch_size = batch_size
        self._num_class = num_class

        self._prefix = f'bytes_{pkt_bytes}_num_{pkt_num}_{model}'
        if not os.path.exists(self._prefix):
            os.makedirs(self._prefix)

    def __new__(cls, *args, **kwargs):
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        logging.set_verbosity(logging.INFO)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

        tf.debugging.set_log_device_placement(False)
        tf.config.set_soft_device_placement(True)
        # tf.config.threading.set_inter_op_parallelism_threads(0)
        # tf.config.threading.set_intra_op_parallelism_threads(0)

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)
        return super().__new__(cls)

    def _parse_sparse_example(self, example_proto):
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
        sparse_features = tf.sparse.slice(sparse_features, start=[0, 0], size=[self._pkt_num, self._pkt_bytes])
        dense_features = tf.sparse.to_dense(sparse_features)
        dense_features = tf.cast(dense_features, tf.float32) / 255.
        return dense_features, labels

    def _generate_ds(self, path_dir, use_cache=False):
        assert os.path.isdir(path_dir)
        ds = tf.data.Dataset.list_files(os.path.join(path_dir, '*.tfrecord'), shuffle=True)
        ds = ds.interleave(
            lambda x: tf.data.TFRecordDataset(x).map(self._parse_sparse_example),
            cycle_length=AUTOTUNE,
            block_length=8,
            num_parallel_calls=AUTOTUNE
        )
        ds = ds.batch(self._batch_size, drop_remainder=False)
        if use_cache:
            ds = ds.cache()
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def _init_input_ds(self):
        self._train_ds = self._generate_ds(self._train_path)
        self._valid_ds = self._generate_ds(self._valid_path)

    @staticmethod
    def _text_cnn_block(x, filters, height, width, data_format='channels_last'):
        x = layers.Conv2D(filters=filters, kernel_size=(height, width),
                          strides=1, data_format=data_format)(x)
        x = layers.BatchNormalization(axis=-1 if data_format == 'channels_last' else 1)(x)
        x = layers.Activation(activation='relu')(x)
        x = tf.reduce_max(x, axis=1, keepdims=False)
        return x

    @staticmethod
    def _conv1d_block(x, filters, data_format='channels_last'):
        x = layers.Conv1D(filters=filters, kernel_size=3, strides=1, padding='same', data_format=data_format)(x)
        x = layers.BatchNormalization(axis=-1 if data_format == 'channels_last' else 1)(x)
        x = layers.Activation(activation='relu')(x)
        return x

    def _enhanced_pbcnn(self):
        x = Input(shape=(self._pkt_num, self._pkt_bytes))
        y = tf.reshape(x, shape=(-1, self._pkt_bytes, 1))
        y = self._conv1d_block(y, filters=64)
        # y = self._conv1d_block(y, filters=64)
        y = layers.MaxPooling1D(pool_size=2, strides=2, padding='same', data_format='channels_last')(y)
        y = self._conv1d_block(y, filters=128)
        # y = self._conv1d_block(y, filters=128)
        y = layers.MaxPooling1D(pool_size=2, strides=2, padding='same', data_format='channels_last')(y)
        filters = 256
        y = self._conv1d_block(y, filters=filters)
        y = tf.reduce_max(y, axis=1, keepdims=False)
        y = tf.reshape(y, shape=(-1, self._pkt_num, filters, 1))
        y1 = self._text_cnn_block(y, filters=256, height=3, width=filters)
        y2 = self._text_cnn_block(y, filters=256, height=4, width=filters)
        y3 = self._text_cnn_block(y, filters=256, height=5, width=filters)
        y = layers.concatenate(inputs=[y1, y2, y3], axis=-1)

        y = layers.Flatten()(y)
        y = layers.Dense(512, activation='relu')(y)
        y = layers.Dense(256, activation='relu')(y)
        # y = layers.Dense(128, activation='relu')(y)
        y = layers.Dense(self._num_class, activation='linear')(y)

        return Model(inputs=x, outputs=y)


    def _pbcnn(self):
        x = Input(shape=(self._pkt_num, self._pkt_bytes))
        y = tf.reshape(x, shape=(-1, self._pkt_num, self._pkt_bytes, 1))
        data_format = 'channels_last'
        y1 = self._text_cnn_block(y, filters=256, height=3, width=self._pkt_bytes)
        y2 = self._text_cnn_block(y, filters=256, height=4, width=self._pkt_bytes)
        y3 = self._text_cnn_block(y, filters=256, height=5, width=self._pkt_bytes)
        y = layers.concatenate(inputs=[y1, y2, y3], axis=-1)
        y = layers.Flatten(data_format=data_format)(y)
        y = layers.Dense(512, activation='relu')(y)
        y = layers.Dense(256, activation='relu')(y)
        # y = layers.Dense(128, activation='relu')(y)
        y = layers.Dense(self._num_class, activation='linear')(y)
        return Model(inputs=x, outputs=y)


    def _init_model(self):
        if self._model_type == 'pbcnn':
            self._model = self._pbcnn()
        else:
            self._model = self._enhanced_pbcnn()
        # self._model.summary()

    def predict(self, model_dir, data_dir=None, digits=6):
        # model = tf.saved_model.load()
        model = K.models.load_model(model_dir)
        if data_dir:
            test_ds = self._generate_ds(data_dir)
        else:
            test_ds = self._generate_ds(self._test_path)
        y_pred, y_true = [], []
        for features, labels in test_ds:
            y_ = model.predict(features)
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

    def init(self):
        self._init_input_ds()
        self._init_model()

    def _init_(self):
        self._optimizer = K.optimizers.Adam()
        # self._loss_func = K.losses.sparse_categorical_crossentropy
        self._loss_func = tf.nn.sparse_softmax_cross_entropy_with_logits
        self._acc_func = K.metrics.sparse_categorical_accuracy

        self._train_losses = []
        self._valid_losses = []
        self._train_acc = []
        self._valid_acc = []

    def _train_step(self, features, labels):
        with tf.GradientTape() as tape:
            y_predict = self._model(features, training=True)
            loss = self._loss_func(labels, y_predict)
            acc_match = self._acc_func(labels, y_predict)

        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
        return loss.numpy().sum(), acc_match.numpy().sum()

    def _test_step(self, features, labels):
        y_predicts = self._model(features, training=False)
        loss = self._loss_func(labels, y_predicts)
        acc_match = self._acc_func(labels, y_predicts)
        return loss.numpy().sum(), acc_match.numpy().sum()

    def train(self,
              epochs,
              log_freq=10,
              valid_freq=300,
              model_dir=f'models_tf',
              history_path='train_history.pkl',
              DEBUG=False):
        history_path = os.path.join(self._prefix, history_path)
        model_dir = os.path.join(self._prefix, model_dir)

        self._init_()
        steps = 1
        try:
            for epoch in range(1 if DEBUG else epochs):
                logging.info(f'Epoch {epoch}/{epochs}')

                sample_count = 0
                total_loss = 0.
                total_match = 0

                for features, labels in self._train_ds:
                    if DEBUG and steps > 300:
                        break

                    loss, match = self._train_step(features, labels)  # batch loss
                    total_loss += loss
                    sample_count += len(features)
                    avg_train_loss = total_loss / sample_count
                    self._train_losses.append(avg_train_loss)

                    total_match += match
                    avg_train_acc = total_match / sample_count
                    self._train_acc.append(avg_train_acc)

                    if log_freq > 0 and steps % log_freq == 0:
                        logging.info('Epoch %d, step %d, avg loss %.6f, avg acc %.6f'
                                     % (epoch, steps, avg_train_loss, avg_train_acc))

                    if valid_freq > 0 and steps % valid_freq == 0:
                        logging.info(f'===> Step: {steps}, evaluating on VALID...')
                        valid_loss, valid_acc = [], []
                        valid_cnt = 0
                        for fs, ls in self._valid_ds:
                            lo, ma = self._test_step(fs, ls)
                            valid_loss.append(lo)
                            valid_acc.append(ma)
                            valid_cnt += len(fs)

                        avg_valid_loss = np.array(valid_loss).sum() / valid_cnt
                        avg_valid_acc = np.array(valid_acc).sum() / valid_cnt
                        logging.info('===> VALID avg loss: %.6f, avg acc: %.6f' % (avg_valid_loss, avg_valid_acc))
                        self._valid_losses.append(avg_valid_loss)
                        self._valid_acc.append(avg_valid_acc)
                    steps += 1
        except Exception as e:
            raise Exception(e)
        finally:
            history = {
                'epoch_steps': steps / epochs,
                'valid_freq': valid_freq,
                'train_loss': self._train_losses,
                'train_acc': self._train_acc,
                'valid_loss': self._valid_losses,
                'valid_acc': self._valid_acc
            }

            with open(history_path, 'wb') as fw:
                pickle.dump(history, fw)

        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)
        tf.saved_model.save(self._model, model_dir)

        logging.info(f'After training {epochs} epochs, '
                     f'save model to {model_dir}, train logs to {history_path}.')


def main(_):
    s = time.time()
    demo = TF(pkt_bytes=256, pkt_num=20, model='pbcnn',
              train_path='E:\\qq_rev\\ids2018\\712_tfrecord\\712_tfrecord\\train',
              valid_path='E:\\qq_rev\\ids2018\\712_tfrecord\\712_tfrecord\\valid',
              test_path='E:\\qq_rev\\ids2018\\712_tfrecord\\712_tfrecord\\test',
              batch_size=1,
              num_class=15)
    # There are two models can be choose, "pbcnn" and "en_pbcnn".
    demo.init()
    # demo.fit(1)
    # print(demo._predict())
    demo.train(epochs=1)
    logging.info(f'cost: {(time.time() - s) / 60} min')


if __name__ == '__main__':
    app.run(main)


