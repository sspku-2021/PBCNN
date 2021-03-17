# -*- coding: utf-8 -*-

import sys
sys.path.append("../../")

"""
PortScan: 100
Bot: 99
XSS: 12
Sql Injection: 5
Brute Force: 116,
Infiltration 1
"""

few_shot_label_list = ["PortScan", "Bot", "Web Attack  XSS", "Web Attack  Sql Injection",
                       "Infiltration", "Web Attack  Brute Force"]

from PBCNN.tools.preprocess import pkt_to_pixel

import pymongo
import numpy as np
from tqdm import tqdm

import random


def add_noise(session_pixel, noise_rate = 0.5):
    noise_idx = random.sample(np.arange(len(session_pixel)).tolist(), int(len(session_pixel) * noise_rate))
    for idx in noise_idx:
        session_pixel[idx] = random.randint(0, 255)
    return session_pixel


def data_transform_with_noise(sessions, height, width, session_length):
    ret = []
    for idx, session in enumerate(sessions):
        if idx == session_length:
            break
        if idx < 4 or idx > len(sessions) - 3:
            session_pixel = add_noise(pkt_to_pixel(session))
            session = bytes2matrix(session_pixel, height, width)
            ret.append(session)
    if len(ret) < session_length:
        for _ in range(session_length - len(ret)):
            ret.append(np.zeros((height, width)))
    ret = np.array(ret, dtype=np.int8).tolist()
    return ret


def data_transform(sessions, height, width, session_length):
    ret = []
    for idx, session in enumerate(sessions):
        if idx == session_length:
            break
        session = bytes2matrix(pkt_to_pixel(session), height, width)
        ret.append(session)
    if len(ret) < session_length:
        for _ in range(session_length - len(ret)):
            ret.append(np.zeros((height, width)))
    ret = np.array(ret, dtype=np.int8).tolist()
    return ret


def bytes2matrix(session, height, width, mask = 0):
    if len(session) > height * width:
        session = session[:height * width]
    else:
        session += [mask] * (height * width - len(session))
    session = np.array(session)
    session = np.reshape(session, newshape=(height, width))
    return session

def _sample(inputs, session_length):
    outputs = inputs[:4]
    if len(inputs) > session_length:
        sample_section = np.arange(4, min(len(inputs)-2, 100)).tolist()
        sample_output = random.sample(sample_section, session_length - 6)
    else:
        sample_section = np.arange(4, len(inputs)-2).tolist()
        sample_output = random.sample(sample_section, int((len(inputs)-6) * 0.7))
    for idx in sample_output:
        outputs.append(inputs[idx])
    return outputs + inputs[-2:]



def agument_(info, nums, height = 32, width = 32, session_length = 24):
    infos = []
    raw_info = {
        "bytes": info["bytes"],
        "label": info["label"],
        "augment": False,
        "source": info["_id"]
       }
    raw_info["length"] = min(session_length, len(info["bytes"]))
    raw_info["bytes"] = data_transform(info["bytes"], height, width, session_length)
    infos.append(raw_info)
    for _ in range(nums):
        _info = {
            "label": info["label"],
            "augment": True,
            "source": info['_id']
        }
        _info_bytes = _sample(info["bytes"], session_length)
        _info["length"] = min(session_length, len(_info_bytes))
        _info["bytes"] = data_transform_with_noise(_info_bytes, height, width, session_length)
        infos.append(_info)

    return infos


def data_augment(collection_in, collection_out):
    collection_out.create_index("label")
    augment_rate = {
        "Web Attack  XSS": 5,
        "Web Attack  Sql Injection": 10,
        "Infiltration": 50,
    }
    labels = collection_in.distinct("label")
    for label in tqdm(labels):
        infos = collection_in.find({"label": label})
        for info in infos:
            info_list = agument_(info, augment_rate.get(label, 0))
            for _info in info_list:
                collection_out.insert(_info)


if __name__ == "__main__":
    host = 'mongodb://localhost:27017/'
    client = pymongo.MongoClient(host)
    db = client["ids_few"]
    collection_in = db["few_shot_raw"]
    collection_out = db["few_shot_aug"]
    data_augment(collection_in, collection_out)
