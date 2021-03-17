# -*- coding: utf-8 -*-
import sys
sys.path.append("../")

import json
import pymongo
import random
import os
import numpy as np

import config
from PBCNN.tools.preprocess import pkt_to_pixel

cat_label_list = ["DoS Hulk", "BENIGN", "DoS GoldenEye", "DoS Slowhttptest",
                  "DoS slowloris", "DDoS", "FTP-Patator"]
few_shot_label_list = ["PortScan", "Bot", "Web Attack  XSS", "Web Attack  Sql Injection",
                       "Infiltration", "Web Attack  Brute Force"]


def data_transform(sessions, height, width, session_length):
    """
    超出session_length的部分被舍弃掉，少于session_length的直接填充0
    """
    ret = []
    for idx, session in enumerate(sessions):
        if idx == session_length:
            break
        session = bytes2matrix(pkt_to_pixel(session), height, width)
        ret.append(session)
    if len(ret) < session_length:
        for _ in range(session_length - len(ret)):
            ret.append(np.zeros((height, width)))
    ret = np.array(ret)
    return ret


def bytes2matrix(session, height, width, mask=0):
    """
    将字节形式的session转换成矩阵形式，过长截断，过短填充0
    """
    if len(session) > height * width:
        session = session[:height * width]
    else:
        session += [mask] * (height * width - len(session))
    session = np.array(session)
    session = np.reshape(session, newshape=(height, width))
    return session


def transform_data(file_in, file_out, height, width, session_length=32):
    """
    将字节流转成矩阵格式
    """
    f_out = open(file_out, "w", encoding="utf-8")
    with open(file_in, "r", encoding="utf-8") as fr:
        for idx, line in enumerate(fr):
            if idx % 100 == 0:
                print("processing {} data".format(idx))
            info = json.loads(line)
            inputs = dict()
            inputs["label"] = info["label"]
            inputs["bytes"] = data_transform(info["bytes_head"], height, width, session_length).tolist()
            inputs["length"] = min(session_length, len(info["bytes_head"]))
            inputs["payloads_size"] = info["payloads_size"]
            inputs["payloads_size_normalize"] = info["payloads_size_normalize"]
            inputs["duration_time"] = info["duration_time"]
            inputs["duration_time_normalize"] = info["duration_time_normalize"]
            inputs = json.dumps(inputs)
            f_out.write(inputs + "\n")
    f_out.close()


def get_label_mapping(file_path, label_mapping_path):
    """
    对各个label做map
    """
    labels_set = set()
    with open(file_path, "r", encoding="utf-8") as fr:
        for line in fr:
            info = json.loads(line)
            if info["label"] != "BENIGN":
                labels_set.add(info["label"])
    label_mapping = {
        "BENIGN": 0
    }
    for idx, label in enumerate(list(labels_set)):
        label_mapping[label] = idx + 1

    with open(label_mapping_path, "w", encoding="utf-8") as fr:
        json.dump(label_mapping, fr)

    return label_mapping


def sample(ids, sample_num):
    """
    partition training set and test set
    """
    random.shuffle(ids)
    return ids[sample_num:], ids[:sample_num]


def all_split(train_file_path, test_file_path):
    """
    a part of the sample is used as the test set and the rest as the training set
    """
    host = 'mongodb://localhost:27017/'
    db_name = 'IDS_2017'
    client = pymongo.MongoClient(host)
    db = client[db_name]
    collections = [collection_name for collection_name in db.list_collection_names() if collection_name.endswith("-Session")]

    train_ids, test_ids = [], []

    for collection_name in collections:
        # labels = db[collection_name].aggregate([{"$group": {"_id": "$label"}}])
        # labels = set(["".join(list(info.values())) for info in labels])
        labels = db[collection_name].distinct("label")
        for label in labels:
            if label == "DIRTY":
                continue
            ids = [session["_id"] for session in db[collection_name].find({"label": label})]
            count = len(ids)
            if count > 10000:
                sample_num = 2000
            elif 5000 < count <= 10000:
                sample_num = 500
            elif 1000 < count <= 5000:
                sample_num = count // 10
            elif 100 < count <= 1000:
                sample_num = count // 5
            else:
                sample_num = count // 2
            tmp_train_ids, tmp_test_ids = sample(ids, sample_num)
            print("label {}, sampling train size: {}, sampling test_size {}.".format(label, len(tmp_train_ids), len(tmp_test_ids)))
            train_ids.extend([(_id, collection_name) for _id in tmp_train_ids])
            test_ids.extend([(_id, collection_name) for _id in tmp_test_ids])

    random.shuffle(train_ids)
    random.shuffle(test_ids)

    with open(train_file_path, "w", encoding="utf-8") as f_out:
        for object_id, collection_name in train_ids:
            session = list(db[collection_name].find({"_id": object_id}))[0]
            del session["_id"]
            session = json.dumps(session)
            f_out.write(session + "\n")

    with open(test_file_path, "w", encoding="utf-8") as f_out:
        for object_id, collection_name in test_ids:
            session = list(db[collection_name].find({"_id": object_id}))[0]
            del session["_id"]
            session = json.dumps(session)
            f_out.write(session + "\n")

    print("There are all {} training data and {} testing data".format(len(train_ids), len(test_ids)))


def data_select(file_in, file_out, labels_list):
    """
    extract the data of the specified label list
    """
    f_out = open(file_out, "w", encoding="utf-8")
    with open(file_in, "r", encoding="utf-8") as fr:
        for line in fr:
            session = json.loads(line.rstrip())
            if session["label"] in labels_list:
                f_out.write(line)
    f_out.close()


if __name__ == "__main__":
    train_file_path = os.path.join(config.data_dir, "train_file_all.txt")
    test_file_path = os.path.join(config.data_dir, "test_file_all.txt")
    # all_split(train_file_path, test_file_path)

    cat_train_file_path = os.path.join(config.data_dir, "train_file_cat.txt")
    cat_test_file_path = os.path.join(config.data_dir, "test_file_cat.txt")
    # data_select(train_file_path, cat_train_file_path, cat_label_list)
    # data_select(test_file_path, cat_test_file_path, cat_label_list)

    few_train_file_path = os.path.join(config.data_dir, "train_file_few.txt")
    few_test_file_path = os.path.join(config.data_dir, "test_file_few.txt")
    # data_select(train_file_path, few_train_file_path, few_shot_label_list)
    # data_select(test_file_path, few_test_file_path, few_shot_label_list)

    height, width = 32, 32
    session_length = 24
    data_dir = os.path.join(config.data_dir, "{} x {} x {}".format(height, width, session_length))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    transform_data(file_in=train_file_path, file_out=os.path.join(data_dir, 'train_file_all.txt'), height=height, width=width, session_length=session_length)
    transform_data(file_in=test_file_path, file_out=os.path.join(data_dir, 'test_file_all.txt'), height=height, width=width, session_length=session_length)

    # height, width = 128, 4
    # session_length = 24
    # data_dir = os.path.join(config.data_dir, "{} x {} x {}".format(height, width, session_length))
    # if not os.path.exists(data_dir):
    #     os.mkdir(data_dir)
    # transform_data(file_in=train_file_path, file_out=os.path.join(data_dir, 'train_file_all.txt'),
    # height=height, width=width, session_length=session_length)
    # transform_data(file_in=test_file_path, file_out=os.path.join(data_dir, 'test_file_all.txt'),
    # height=height, width=width, session_length=session_length)
    label_mapping_path = os.path.join(config.cache_dir, "label_mapping_all.pkl")
    get_label_mapping(test_file_path, label_mapping_path)












