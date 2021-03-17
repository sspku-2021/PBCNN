# -*- coding: utf-8 -*-


"""
PortScan: 100
Bot: 99
XSS: 12
Sql Injection: 5
Brute Force: 116,
Infiltration 1
"""

import sys
sys.path.append("../../")

import config

import pymongo
import random
from itertools import combinations
from tqdm import tqdm
import copy
import json
import os


def split(collect_in, collect_train, collect_val):
    collect_train.create_index("label")
    collect_val.create_index("label")

    ids = list(collect_in.distinct("_id"))
    random.shuffle(ids)
    thres = int(len(ids) * 0.9)

    for idx, object_id in enumerate(ids):
        session = list(collect_in.find({"_id": object_id}))[0]
        del session["_id"]
        if idx < thres:
            collect_train.insert(session)
        else:
            collect_val.insert(session)


def construct_pair(info_1, info_2, label):
    try:
        return {
            "session_1": info_1["bytes"],
            "session_2": info_2["bytes"],
            "length_1": info_1["length"],
            "length_2": info_2["length"],
            "label": label,
            "session1_label": info_1["label"],
            "session2_label": info_2["label"],
            "is_source": info_1["source"] == info_2["source"]
        }
    except:
         return {
            "session_1": info_1["bytes"],
            "session_2": info_2["bytes"],
            "length_1": info_1["length"],
            "length_2": info_2["length"],
            "label": label,
            "session1_label": info_1["label"],
            "session2_label": info_2["label"],
            "is_source": False,
        }


def prepare_data(collection, file_out):
    true_sample_rate = {
        "Web Attack  XSS": 0.2,
        "Web Attack  Sql Injection": 0.3,
        "Infiltration": 0.5,
    }

    outputs = []
    labels = collection.distinct("label")
    for label in tqdm(labels):
        ids = [info["_id"] for info in collection.find({"label": label})]
        random.shuffle(ids)
        # 构造正样本
        if label in true_sample_rate:
            source_pair_set = set()
        true_pairs = list(combinations(ids, 2))
        true_numbers = 0
        for pair in true_pairs:
            info_1 = collection.find_one({"_id": pair[0]})
            info_2 = collection.find_one({"_id": pair[1]})
            if label in true_sample_rate:
                if info_1["source"] == info_2["source"]:
                    sample_rate = true_sample_rate.get(label)
                    if sample_rate != None:
                        if random.uniform(0, 1) < sample_rate:
                            true_numbers += 1
                            outputs.append((pair[0], pair[1], 1))
                            source_pair_set.add((info_1["source"], info_2["source"]))
                elif (info_1["source"], info_2["source"]) in source_pair_set or (info_2["source"], info_1["source"]) in source_pair_set:
                    if random.uniform(0, 1) < 0.3:
                        true_numbers += 1
                        outputs.append((pair[0], pair[1], 1))
                else:
                    true_numbers += 1
                    outputs.append((pair[0], pair[1], 1))
                    source_pair_set.add((info_1["source"], info_2["source"]))
            else:
                true_numbers += 1
                outputs.append((pair[0], pair[1], 1))
        print("There are {} true pairs in label {}.".format(true_numbers, label))

        # 构造负样本 1:1.5
        neg_labels = copy.deepcopy(labels)
        neg_labels.remove(label)
        neg_ids = []
        for neg_label in neg_labels:
            neg_ids.extend([info["_id"] for info in collection.find({"label": neg_label})])
        random.shuffle(neg_ids)
        neg_numbers = int(true_numbers * 1.5)
        neg_number_per = neg_numbers // len(ids)
        for true_id in ids:
            neg_sample_ids = random.sample(neg_ids, neg_number_per)
            for neg_id in neg_sample_ids:
                outputs.append((true_id, neg_id, 0))

    random.shuffle(outputs)
    f_out = open(file_out, "w", encoding="utf-8")
    for pair in tqdm(outputs):
        info_0 = collection.find_one({"_id": pair[0]})
        info_1 = collection.find_one({"_id": pair[1]})
        pair = construct_pair(info_0, info_1, pair[2])
        pair = json.dumps(pair)
        f_out.write(pair + "\n")
    f_out.close()


if __name__ == "__main__":
    host = 'mongodb://localhost:27017/'
    client = pymongo.MongoClient(host)
    db = client["ids_few"]
    collection_in = db["few_shot_aug"]
    collection_train = db["few_shot_aug_train"]
    collection_val = db["few_shot_aug_val"]
    split(collection_in, collection_train, collection_val)
    few_shot_dir = os.path.join(os.path.join("../", config.data_dir), "few_shot")
    if not os.path.exists(few_shot_dir):
        os.mkdir(few_shot_dir)
    train_file_path = os.path.join(few_shot_dir, "train_noise.txt")
    val_file_path = os.path.join(few_shot_dir, "val_noise.txt")
    prepare_data(collection_train, train_file_path)
    prepare_data(collection_val, val_file_path)









