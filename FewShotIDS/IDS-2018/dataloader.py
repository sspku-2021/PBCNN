import json
import os
import numpy as np
import random
import time

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


class Dataloader():
    def __init__(self, batch_size, nb_per_classes):
        self.dataset_path = "data"
        self.label_mapping = ['ftp-bruteforce', 'ddos-hoic', 'dos-goldeneye', 'ddos-loic-http',
                              'sql-injection', 'dos-hulk', 'bot', 'ssh-bruteforce', 'bruteforce-xss',
                              'dos-slowhttptest', 'bruteforce-web', 'dos-slowloris', 'benign', 'ddos-loic-udp',
                              'infiltration']
        self.batch_size = batch_size
        self.nb_per_classes = nb_per_classes
        self.load_dataset()

    def load_dataset(self):
        print("loading dataset")
        self.train_dataset = self._load_dataset(os.path.join(self.dataset_path, "train.json"))
        self.val_dataset = self._load_dataset(os.path.join(self.dataset_path, "valid.json"))
        self.test_dataset = self._load_dataset(os.path.join(self.dataset_path, "test.json"))

    def _load_dataset(self, path):
        """
        return:dataset,一个字典，key为label，value为对应的pkts列表
        """
        dataset = {}
        with open(path, "r", encoding="utf-8") as fr:
            for info in fr:
                info = json.loads(info)
                if dataset.get(self.label_mapping[info["label"]], None) is None:
                    dataset[self.label_mapping[info["label"]]] = []
                dataset[self.label_mapping[info["label"]]].append(np.array(info["pkts"])/255.0)

        if "train" in path:
            for key, val in dataset.items():
                random.shuffle(val)
                dataset[key] = val[:min(len(val), self.nb_per_classes)]
        return dataset

    def _construct_pairs(self, batch_pkts, is_one_shot_task):
        nb_pairs = len(batch_pkts)//2
        pairs_of_pkts = [np.zeros((nb_pairs, 20, 256)) for _ in range(2)]
        labels = np.zeros((nb_pairs, 1))

        for idx in range(nb_pairs):
            pairs_of_pkts[0][idx, :, :] = batch_pkts[idx * 2]
            pairs_of_pkts[1][idx, :, :] = batch_pkts[idx * 2 + 1]

            if not is_one_shot_task:
                if (idx + 1) % 2 == 0:
                    labels[idx] = 0
                else:
                    labels[idx] = 1
            else:
                if idx == 0:
                    labels[idx] = 1
                else:
                    labels[idx] = 0

        if not is_one_shot_task:
            # 打乱labels
            random_permutation_index = np.random.permutation(nb_pairs)
            labels = labels[random_permutation_index]
            pairs_of_pkts[0][:, :, :] = pairs_of_pkts[0][random_permutation_index, :, :]
            pairs_of_pkts[1][:, :, :] = pairs_of_pkts[1][random_permutation_index, :, :]

        return pairs_of_pkts, labels

    def get_train_batch(self):
        """
        选中的label：3 pkts
        另一个label：1 pkt
        """
        available_labels = list(self.train_dataset.keys())
        number_of_labels = len(available_labels)
        selected_labels = [self.label_mapping[random.randint(0, number_of_labels-1)] for _ in range(self.batch_size)]
        batch_pkts = []
        for idx, label in enumerate(selected_labels):
            # random select 3 pkts from the same label
            pkts_indexes = random.sample(range(0, len(self.train_dataset[label])), 3)
            batch_pkts.append(self.train_dataset[label][pkts_indexes[0]])
            batch_pkts.append(self.train_dataset[label][pkts_indexes[1]])
            batch_pkts.append(self.train_dataset[label][pkts_indexes[2]])

            different_labels = available_labels[:]
            different_labels.remove(label)
            different_label = random.sample(different_labels, 1)[0]
            different_pkt_index = random.sample(range(0, len(self.train_dataset[different_label])), 1)[0]
            batch_pkts.append(self.train_dataset[different_label][different_pkt_index])

        pkts, labels = self._construct_pairs(batch_pkts, is_one_shot_task=False)
        return pkts, labels

    def get_one_shot_batch(self, label, dataset, support_set_size):
        available_labels = list(dataset.keys())
        number_of_labels = len(available_labels)

        batch_pkts = []
        pkt_indexes = random.sample(range(0, len(dataset[label])), 2)
        batch_pkts.append(dataset[label][pkt_indexes[0]])
        batch_pkts.append(dataset[label][pkt_indexes[1]])

        if support_set_size == -1:
            support_set_size = number_of_labels
        support_set_size = min(support_set_size, number_of_labels)

        different_labels = available_labels[:]
        different_labels.remove(label)

        support_labels = random.sample(different_labels, support_set_size-1)
        for different_label in support_labels:
            batch_pkts.append(dataset[label][pkt_indexes[0]])
            batch_pkts.append(dataset[different_label][random.sample(range(0, len(dataset[different_label])), 1)[0]])

        pkts, labels = self._construct_pairs(batch_pkts, is_one_shot_task=True)
        return pkts, [label] + support_labels  # ？？？

    def one_shot_test(self, model, support_set_size, number_of_tasks_per_label, is_validation, file_out):
        if is_validation:
            dataset = self.val_dataset
            print('\nMaking One Shot Task on validation dataset')
        else:
            dataset = self.test_dataset
            print('\nMaking One Shot Task on test dataset')

        cnt = 0
        if is_validation:
            mean_global_accuracy = 0
            for label in self.label_mapping:
                label_accuracy = 0
                for _ in range(min(number_of_tasks_per_label, len(dataset[label]))):
                    pair_of_pkts, _ = self.get_one_shot_batch(label, dataset, support_set_size)
                    probabilities = model.predict_on_batch(pair_of_pkts)
                    if np.argmax(probabilities) == 0:
                        acc = 1.
                    else:
                        acc = 0.
                    label_accuracy += acc
                    mean_global_accuracy += acc
                    cnt += 1

                label_accuracy /= min(number_of_tasks_per_label, len(dataset[label]))
                print("{} accuracy: {}".format(label, label_accuracy))
                file_out.write(label + "," + str(label_accuracy) + "\n")

            mean_global_accuracy /= cnt
            print("global accuracy: {}".format(mean_global_accuracy))
            file_out.write("global accuracy" + "," + str(mean_global_accuracy) + "\n")
            return mean_global_accuracy
        else:
            preds = []
            labels = []
            for label in self.label_mapping:
                for _ in range(min(number_of_tasks_per_label, len(dataset[label]))):
                    labels.append(self.label_mapping.index(label))
                    pair_of_pkts, _labels = self.get_one_shot_batch(label, dataset, support_set_size)
                    probabilities = model.predict_on_batch(pair_of_pkts)
                    pred = np.argmax(probabilities)
                    preds.append(self.label_mapping.index(_labels[pred]))

            report = classification_report(labels, preds, target_names=self.label_mapping, digits=4)
            print(report)
            file_out.write(report)
            mean_global_accuracy = accuracy_score(labels, preds)
            print("global accuracy: {}".format(mean_global_accuracy))
            file_out.write("global accuracy" + "," + str(mean_global_accuracy) + "\n")
            return mean_global_accuracy

    def one_shot_test_time(self, model, support_set_size, number_of_tasks_per_label):
        dataset = self.test_dataset
        sample_time = 0
        model_time = 0

        for label in self.label_mapping:
            for _ in range(min(number_of_tasks_per_label, len(dataset[label]))):
                start_time = time.time()
                pair_of_pkts, _labels = self.get_one_shot_batch(label, dataset, support_set_size)
                end_time = time.time()
                sample_time += (end_time - start_time)

                start_time = time.time()
                probabilities = model.predict_on_batch(pair_of_pkts)
                np.argmax(probabilities)
                end_time = time.time()
                model_time += (end_time - start_time)

        print("sample time: ", sample_time)
        print("model time: ", model_time)


if __name__ == "__main__":
    dataloader = Dataloader(batch_size=32)
    # dataloader.get_train_batch()
    dataloader.one_shot_test(None, 5, 10, True)


