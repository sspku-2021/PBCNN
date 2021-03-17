import tensorflow as tf
import numpy as np
import os

from tensorflow.keras import Input
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras

from dataloader import Dataloader
from utils import cosine_distance, euclidean_distance, cos_dist_output_shape, eucl_dist_output_shape


class PBCNN(object):
    def __init__(self, batch_size, learning_rate):
        self._pkt_num = 20
        self._pkt_bytes = 256
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._num_class = 15
        self.construct_pbcnn()

    @staticmethod
    def _text_cnn_block(x, filters, height, width, data_format='channels_last'):
        x = layers.Conv2D(filters=filters, kernel_size=(height, width), strides=1, data_format=data_format)(x)
        x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5)(x)
        x = layers.Activation(activation='relu')(x)
        x = tf.reduce_max(x, axis=1, keepdims=False)
        return x

    def _cnn_model(self, x):
        y = tf.reshape(x, shape=(-1, self._pkt_num, self._pkt_bytes, 1))
        data_format = 'channels_last'
        y1 = self._text_cnn_block(y, filters=256, height=3, width=self._pkt_bytes)
        y2 = self._text_cnn_block(y, filters=256, height=4, width=self._pkt_bytes)
        y3 = self._text_cnn_block(y, filters=256, height=5, width=self._pkt_bytes)
        y = layers.concatenate(inputs=[y1, y2, y3], axis=-1)

        y = layers.Flatten(data_format=data_format)(y)
        return y

    def construct_pbcnn(self):
        x = Input(shape=(self._pkt_num, self._pkt_bytes))
        y = self._cnn_model(x)
        y = layers.Dense(512, activation='relu')(y)
        y = layers.Dense(256, activation='relu')(y)
        y = layers.Dense(self._num_class, activation='linear')(y)
        self.model = Model(inputs=[x], outputs=y)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=[keras.metrics.SparseCategoricalAccuracy(name='mean/acc'),
                                    keras.metrics.SparseCategoricalCrossentropy(name='mean/loss', from_logits=True)])

    def train(self, nb_per_classes):
        def load_data(path, nb_per_classes=None):
            import json
            import random
            data = []
            with open(path, "r", encoding="utf-8") as fr:
                for line in fr:
                    info = json.loads(line)
                    data.append((np.array(info["pkts"])/255., info["label"]))
            random.shuffle(data)
            if nb_per_classes is not None:
                data = data[:nb_per_classes]
            X = [info[0] for info in data]
            y = [info[1] for info in data]
            return np.array(X), np.array(y)

        label_mapping = ['ftp-bruteforce', 'ddos-hoic', 'dos-goldeneye', 'ddos-loic-http', 'sql-injection',
                         'dos-hulk', 'bot', 'ssh-bruteforce', 'bruteforce-xss', 'dos-slowhttptest',
                         'bruteforce-web', 'dos-slowloris', 'benign', 'ddos-loic-udp', 'infiltration']

        X_train, y_train = load_data(os.path.join("data", "train.json"), nb_per_classes)
        X_valid, y_valid = load_data(os.path.join("data", "valid.json"))
        self.model.fit(
            X_train, y_train,
            epochs=300,
            batch_size=min(nb_per_classes, 64),
            validation_data=(X_valid, y_valid),
            verbose=2,
            shuffle=False,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=2)]
        )

        self.model.save_weights(os.path.join("models", "pbcnn.h5"))

        # X_test, y_test = load_data(os.path.join("./data", "test.json"))
        # y_preds = self.model.predict(X_test)
        #
        # y_preds = np.argmax(y_preds, axis=-1)
        #
        # from sklearn.metrics import classification_report
        # from sklearn.metrics import accuracy_score
        #
        # print(classification_report(y_test, y_preds, target_names=label_mapping, digits=4))
        # print(accuracy_score(y_test, y_preds))


if __name__ == "__main__":
    pbcnn = PBCNN(batch_size=32, learning_rate=3e-3)
    pbcnn.train(nb_per_classes=10)


class Siamese(object):
    def __init__(self, batch_size, learning_rate, tensorboard_log_path, nb_per_classes):
        self._pkt_num = 20
        self._pkt_bytes = 256
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.nb_per_classes = nb_per_classes
        self.data_loader = Dataloader(self.batch_size, nb_per_classes)
        # self.summary_writer = tf.summary.FileWriter(tensorboard_log_path)
        self.construct_simemse_architecture()

    @staticmethod
    def _text_cnn_block(x, filters, height, width, data_format='channels_last'):
        x = layers.Conv2D(filters=filters, kernel_size=(height, width), strides=1, data_format=data_format)(x)
        x = layers.BatchNormalization(axis=-1, epsilon=1.001e-5)(x)
        x = layers.Activation(activation='relu')(x)
        x = tf.reduce_max(x, axis=1, keepdims=False)
        return x

    def _cnn_model(self, x):
        y = tf.reshape(x, shape=(-1, self._pkt_num, self._pkt_bytes, 1))
        data_format = 'channels_last'
        y1 = self._text_cnn_block(y, filters=256, height=3, width=self._pkt_bytes)
        y2 = self._text_cnn_block(y, filters=256, height=4, width=self._pkt_bytes)
        y3 = self._text_cnn_block(y, filters=256, height=5, width=self._pkt_bytes)
        y = layers.concatenate(inputs=[y1, y2, y3], axis=-1)

        y = layers.Flatten(data_format=data_format)(y)
        return y

    def construct_simemse_architecture(self):
        x1 = Input(shape=(self._pkt_num, self._pkt_bytes))
        x2 = Input(shape=(self._pkt_num, self._pkt_bytes))

        y1 = self._cnn_model(x1)
        y2 = self._cnn_model(x2)

        # l1_distance_layer = layers.Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
        # l1_distance = l1_distance_layer([y1, y2])
        cos_similarity = layers.Lambda(cosine_distance, output_shape=cos_dist_output_shape)([y1, y2])
        l2_distance = layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([y1, y2])

        output = layers.concatenate(inputs=[y1, y2, cos_similarity, l2_distance], axis=-1)
        hidden = layers.Dense(512, activation="relu")(output)
        hidden = layers.Dense(256, activation="relu")(hidden)

        prediction = layers.Dense(1, activation="sigmoid")(hidden)
        self.model = Model(inputs=[x1, x2], outputs=prediction)
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss='binary_crossentropy', metrics=['binary_accuracy'])

    def _write_logs_to_tensorboard(self, current_iteration, train_losses, train_accuracies,
                                   validation_accuracy, evaluate_each):
        """ Writes the logs to a tensorflow log file
        This allows us to see the loss curves and the metrics in tensorboard.
        If we wrote every iteration, the training process would be slow, so
        instead we write the logs every evaluate_each iteration.
        Arguments:
            current_iteration: iteration to be written in the log file
            train_losses: contains the train losses from the last evaluate_each
                iterations.
            train_accuracies: the same as train_losses but with the accuracies
                in the training set.
            validation_accuracy: accuracy in the current one-shot task in the
                validation set
            evaluate each: number of iterations defined to evaluate the one-shot
                tasks.
        """

        # summary = tf.Summary()

        # Write to log file the values from the last evaluate_every iterations
        # for index in range(0, evaluate_each):
        #     value = summary.value.add()
        #     value.simple_value = train_losses[index]
        #     value.tag = 'Train Loss'
        #
        #     value = summary.value.add()
        #     value.simple_value = train_accuracies[index]
        #     value.tag = 'Train Accuracy'
        #
        #     if index == (evaluate_each - 1):
        #         value = summary.value.add()
        #         value.simple_value = validation_accuracy
        #         value.tag = 'One-Shot Validation Accuracy'
        #
        #     self.summary_writer.add_summary(
        #         summary, current_iteration - evaluate_each + index + 1)
        #     self.summary_writer.flush()

    def train_siamese_network(self, number_of_iterations, support_set_size, evaluate_each, log_path, model_name):
        train_losses = np.zeros(shape=(evaluate_each))
        train_accuracies = np.zeros(shape=(evaluate_each))
        count = 0

        best_validation_accuracy = 0.0
        best_accuracy_iteration = 0

        train_out = open(log_path + ".train.csv", "w", encoding="utf-8")
        val_out = open(log_path + ".valid.csv", "w", encoding="utf-8")

        for iteration in range(number_of_iterations):
            pair_pkts, labels = self.data_loader.get_train_batch()
            train_loss, train_accuracy = self.model.train_on_batch(pair_pkts, labels)
            train_out.write(str(iteration) + "," + str(train_loss) + "," + str(train_accuracy) + "\n")

            train_losses[count] = train_loss
            train_accuracies[count] = train_accuracy

            count += 1
            print('Iteration %d/%d: Train loss: %f, Train Accuracy: %f' %
                  (iteration + 1, number_of_iterations, train_loss, train_accuracy))

            if (iteration + 1) % evaluate_each == 0:
                number_of_runs_per_label = 50
                val_out.write(str(iteration) + ":" + "\n")
                validation_accuracy = self.data_loader.one_shot_test(self.model, support_set_size, number_of_runs_per_label, is_validation=True, file_out=val_out)

                self._write_logs_to_tensorboard(
                    iteration, train_losses, train_accuracies,
                    validation_accuracy, evaluate_each)
                count = 0

                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_accuracy_iteration = iteration
                    if not os.path.exists("models"):
                        os.makedirs("models")
                    self.model.save_weights(os.path.join("models", model_name))

            if iteration - best_accuracy_iteration > 1000:
                print(
                    'Early Stopping: validation accuracy did not increase for 500 iterations')
                print('Best Validation Accuracy = ' +
                      str(best_validation_accuracy))
                print('Validation Accuracy = ' + str(best_validation_accuracy))
                break

        train_out.close()
        val_out.close()

        print('Trained Ended!')
        return best_validation_accuracy

    







