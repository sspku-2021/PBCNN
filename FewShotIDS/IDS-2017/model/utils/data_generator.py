# -*- coding: utf-8 -*- #

import numpy as np
import json

import tensorflow as tf
# def data_transform(sessions, height, width, session_length):
#     ret = []
#     for idx, session in enumerate(sessions):
#         if idx == session_length:
#             break
#         session = bytes2matrix(session, height, width)
#         ret.append(session)
#     if len(ret) < session_length:
#         for _ in range(session_length - len(ret)):
#             ret.append(np.zeros((height, width)))
#     ret = np.array(ret)
#     # ret.shape = [session_length, height, width] => [height, width, session_length]
#     ret = np.reshape(ret, newshape=(ret.shape[1], ret.shape[2], ret.shape[0]))
#     return ret
#
# def bytes2matrix(session, height, width):
#     assert len(session) % 2 == 0
#     matrix = []
#     session = [session[i: i+2] for i in range(0, len(session), 2)]
#     bytes_per_row = width // 8
#     bound = min(len(session) // bytes_per_row , height)
#     for i in range(0, bound, bytes_per_row):
#         matrix.append(concat_hex2bin(session[i: min(i+bytes_per_row, len(session))], bytes_per_row))
#     if len(matrix) < height:
#         matrix += [[0] * width for _ in range(height - len(matrix))]
#     assert len(matrix) == height
#     return matrix
#
# def concat_hex2bin(sessions, bytes_per_row):
#     ret = []
#     for session in sessions:
#         ret += hex2bin(session)
#     ret += [0] * 8 * (bytes_per_row - len(sessions))
#     return ret
#
# def hex2bin(hex_str):
#     bin_str = bin(int(hex_str, 16))[2:]
#     return left_padding(bin_str)
#
# def left_padding(bin_str):
#     bin_str = [0] * (8 - len(bin_str)) + list(map(int, bin_str))
#     return bin_str

def input_fn(file_name, label_mapping, batch_size, num_epochs,
             is_label=True, is_length=False, is_size=True, is_time=True):
    def decode_size(line):
        line = line.rstrip()
        line = json.loads(line)
        return line["payloads_size"]

    def decode_time(line):
        line = line.rstrip()
        line = json.loads(line)
        return line["duration_time"]

    def decode_length(line):
        line = line.rstrip()
        line = json.loads(line)
        return line["length"]

    def decode_label(line):
        line = line.rstrip()
        line = json.loads(line)
        return label_mapping[line["label"]]

    def decode_bytes(line):
        line = line.rstrip()
        line = json.loads(line)
        return np.array(line["bytes"], dtype=np.float32)

    def decode_line(line):
        info = dict()
        info["bytes"] = tf.py_func(decode_bytes, [line], [tf.float32])
        if is_size:
            info["size"] = tf.py_func(decode_size, [line], [tf.int64])
        if is_time:
            info["time"] = tf.py_func(decode_time, [line], [tf.double])
        if is_label:
            info["label"] = np.array(tf.py_func(decode_label, [line], [tf.int64]))
        if is_length:
            info["length"] = tf.py_func(decode_length, [line], [tf.int64])
        return info

    dataset = tf.data.TextLineDataset(file_name).map(decode_line, num_parallel_calls=4).prefetch(batch_size * 4)
    dataset = dataset.repeat(num_epochs)

    dataset = dataset.batch(batch_size=batch_size,
                            drop_remainder=True)

    iterator = dataset.make_one_shot_iterator()
    batch_features = iterator.get_next()
    return batch_features


class PairBatcher():
    def __init__(self):
        self.session1 = []
        self.session2 = []
        self.length1 = []
        self.length2 = []
        self.label = []

class DataGeneratorPair():
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

    def batch_generator(self):
        batch = []
        with open(self.path, "r", encoding="utf-8") as fr:
            for idx, line in enumerate(fr):
                batch.append(json.loads(line))
                if len(batch) == self.batch_size:
                    yield self.createBatch(batch)
                    batch = []
        if len(batch) != 0:
            yield self.createBatch(batch)
        yield None
        return

    def createBatch(self, samples):
        batcher = PairBatcher()
        sessions1 = []
        sessions2 = []
        length1 = []
        length2 = []
        labels = []

        for info in samples:
            sessions1.append(info["session_1"])
            sessions2.append(info["session_2"])
            length1.append(info["length_1"])
            length2.append(info["length_2"])
            labels.append(info["label"])

        batcher.session1 = sessions1
        batcher.session2 = sessions2
        batcher.length1 = length1
        batcher.length2 = length2
        batcher.label = labels
        return batcher



class Batcher():
    def __init__(self):
        self.session = []
        self.session_length = []
        self.label = []
        self.duration_time = []
        self.session_size = []

class DataGenerator_Memory():
    def __init__(self, path, batch_size, is_label = True, is_time = False, is_size = False, is_length = False):
        self.path = path
        self.batch_size = batch_size


        self.is_label = is_label
        self.is_time = is_time
        self.is_size = is_size
        self.is_length = is_length

        self.raw_data = []
        cnt = 0
        with open(self.path, "r", encoding="utf-8") as fr:
            for line in fr:
                cnt += 1
                self.raw_data.append(json.loads(line))
        print("There are all {} data.".format(cnt))

    def batch_generator(self, label_mapping=None):
        for i in range(0, len(self.raw_data), self.batch_size):
            yield self.createBatch(self.raw_data[i: min(i + self.batch_size, len(self.raw_data))], label_mapping)
        yield None
        return

    def createBatch(self, samples, label_mapping = None):
        batcher = Batcher()
        sessions = []

        if self.is_length:
            session_lengths = []
        if self.is_label:
            labels = []
        if self.is_size:
            session_sizes = []
        if self.is_time:
            duration_times = []

        for info in samples:
            sessions.append(info["bytes"])
            if self.is_length:
                session_lengths.append(info["length"])
            if self.is_label:
                labels.append(label_mapping[info["label"]])
            if self.is_time:
                duration_times.append(info["duration_time_normalize"])
            if self.is_size:
                session_sizes.append(info["payloads_size_normalize"])

        # sessions = np.array(sessions, dtype=np.float32)
        #sessions = np.reshape(sessions, newshape=(sessions.shape[0], sessions.shape[3], sessions.shape[1], sessions.shape[2]))
        batcher.session = sessions
        if self.is_label:
            batcher.label = np.array(labels)
        if self.is_length:
            batcher.session_length = session_lengths
        if self.is_size:
            batcher.session_size = session_sizes
        if self.is_time:
            batcher.duration_time = duration_times
        return batcher


class DataGenerator():

    def __init__(self, path, batch_size, is_label = True, is_time = False, is_size = False, is_length = False):
        self.path = path
        self.batch_size = batch_size


        self.is_label = is_label
        self.is_time = is_time
        self.is_size = is_size
        self.is_length = is_length

        self.raw_data = []


    def batch_generator(self, label_mapping = None):
        batch = []
        with open(self.path, "r", encoding="utf-8") as fr:
            for idx, line in enumerate(fr):
                batch.append(json.loads(line))
                if len(batch) == self.batch_size:
                    yield self.createBatch(batch, label_mapping)
                    batch = []
        if len(batch) != 0:
            yield self.createBatch(batch, label_mapping)
        yield None
        return

    def createBatch(self, samples, label_mapping = None):
        batcher = Batcher()
        sessions = []

        if self.is_length:
            session_lengths = []
        if self.is_label:
            labels = []
        if self.is_size:
            session_sizes = []
        if self.is_time:
            duration_times = []

        for info in samples:
            sessions.append(info["bytes"])
            if self.is_length:
                session_lengths.append(info["length"])
            if self.is_label:
                labels.append(label_mapping[info["label"]])
            if self.is_time:
                duration_times.append(info["duration_time_normalize"])
            if self.is_size:
                session_sizes.append(info["payloads_size_normalize"])

        # sessions = np.array(sessions, dtype=np.float32)
        #sessions = np.reshape(sessions, newshape=(sessions.shape[0], sessions.shape[3], sessions.shape[1], sessions.shape[2]))
        batcher.session = sessions
        if self.is_label:
            batcher.label = np.array(labels)
        if self.is_length:
            batcher.session_length = session_lengths
        if self.is_size:
            batcher.session_size = session_sizes
        if self.is_time:
            batcher.duration_time = duration_times
        return batcher





