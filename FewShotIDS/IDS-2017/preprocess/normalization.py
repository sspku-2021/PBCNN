# -*- coding: utf-8 -*-
import sys
sys.path.append("../")


import pymongo
import os
import pickle
import math
from pymongo.database import Database
from PBCNN.tools.preprocess import standardzation
import config
from tqdm import tqdm


def get_mu_sigma(db: Database, path: str = config.stardard_cache_path):
    if os.path.exists(path):
        with open(path, "rb") as pkl:
            return pickle.load(pkl)

    time_mu = 0
    size_mu = 0
    count = 0
    collections = [collection_name for collection_name in db.list_collection_names() if collection_name.endswith("Session")]

    for collection_name in tqdm(collections):
        collection = db[collection_name]
        for info in collection.find():
            time_mu += info["duration_time"]
            size_mu += info["payloads_size"]
            count += 1
    time_mu /= count
    size_mu /= count

    time_sigma = 0
    size_sigma = 0
    for collection_name in tqdm(collections):
        collection = db[collection_name]
        for info in collection.find():
            time_sigma += (info["duration_time"] - time_mu) ** 2
            size_sigma += (info["payloads_size"] - size_mu) ** 2
    time_sigma = math.sqrt(time_sigma / count)
    size_sigma = math.sqrt(size_sigma / count)

    with open(path, "wb") as pkl:
        pickle.dump((time_mu, size_mu, time_sigma, size_sigma), pkl)

    return time_mu, size_mu, time_sigma, size_sigma


def normalize(collection, params, normalize_fn=standardzation):
    time_mu, size_mu, time_sigma, size_sigma = params
    for info in collection.find():
        info["duration_time_normalize"] = normalize_fn(info["duration_time"], time_mu, time_sigma)
        info["payloads_size_normalize"] = normalize_fn(info["payloads_size"], size_mu, size_sigma)
        collection.update_one({"_id": info["_id"]}, {'$set': info})


if __name__ == "__main__":
    host = 'mongodb://localhost:27017/'
    db_name = 'IDS_2017'
    client = pymongo.MongoClient(host)
    db = client[db_name]
    params = get_mu_sigma(db)

    collections = [collection_name for collection_name in db.list_collection_names() if collection_name.endswith("-Session")]
    for collection_name in tqdm(collections):
        collection = db[collection_name]
        normalize(collection, params, standardzation)












