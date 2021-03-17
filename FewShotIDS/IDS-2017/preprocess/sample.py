import sys
sys.path.append("../")


import pymongo
import json
import random
import os


import config
from PBCNN.tools.preprocess import bytes2str, str2bytes


def get_count(db, collection_name, thres):
    """
    calculate the number of "BENIGN" and "ATTACK"
    """
    print("getting count...")
    labels = db[collection_name].aggregate([{"$group": {"_id": "$label"}}])
    labels = set(["".join(list(info.values())) for info in labels])
    count = dict()
    for label in labels:
        if label != "BENIGN":
            count["ATTACK"] = count.get("ATTACK", 0) + db[collection_name].find({"label": label}, {"_id": 0}).count()
        else:
            sessions = db[collection_name].find({"label": label}, {"_id": 0})
            for session in sessions:
                if thres[0] <= len(session["bytes"]) <= thres[1]:
                    count["BENIGN"] = count.get("BENIGN", 0) + 1
    return count


def split(session, thres=30):
    """
    split session
    """
    def get_property(session_in):
        session_out = dict()
        for key, val in session_in.items():
            if key == "bytes":
                session_out[key] = []
            else:
                session_out[key] = val
        return session_out

    def trans_str2bytes(session):
        for i in range(len(session["bytes"])):
            session["bytes"][i] = str2bytes(session["bytes"][i])
        return session

    def trans_bytes2str(session):
        for i in range(len(session["bytes"])):
            session["bytes"][i] = bytes2str(session["bytes"][i])

    session = trans_str2bytes(session)

    outputs = []
    for i in range(len(session["bytes"])):
        if i == 0:
            output = get_property(session)
        else:
            if session["bytes"][i].time - output["bytes"][-1].time >= thres:
                outputs.append(trans_bytes2str(output))
                output = get_property(session)
        output["bytes"].append(session["bytes"][i])

    if len(output["bytes"]) != 0:
        outputs.append(trans_bytes2str(output))

    return outputs


def sample(db, collection_name, thres, count, f_out, sample_rate):
    # """
    # 对于标签为"BENIGN"的sessions,按照特定比例采样，且session的byte长度需满足一定范围，否则舍弃该session
    # 对于标签为"ATTACK"的sessions，按照特定大小做切割
    # """
    print("sampling")
    labels = db[collection_name].aggregate([{"$group": {"_id": "$label"}}])
    labels = set(["".join(list(info.values())) for info in labels])
    for label in labels:
        sessions = db[collection_name].find({"label": label}, {"_id": 0})
        if label == "BENIGN":
            for session in sessions:
                if thres[0] <= len(session["bytes"]) <= thres[1]:
                    if random.uniform(0, count["BENIGN"]) <= count["ATTACK"] * sample_rate:
                        session = json.dumps(session)
                        f_out.write(session + "\n")
        else:
            for session in sessions:
                if len(session["bytes"]) < thres[1]:
                    session = json.dumps(session)
                    f_out.write(session + "\n")
                else:
                    session_ = split(session)
                    for session in session_:
                        session = json.dumps(session)
                        f_out.write(session + "\n")


if __name__ == "__main__":
    host = 'mongodb://localhost:27017/'
    db_name = 'IDS_2017'
    client = pymongo.MongoClient(host)
    db = client[db_name]
    f_out = open(os.path.join(config.data_dir, "raw_data.txt"), "w", encoding="utf-8")
    for collection_name in db.list_collection_names():
        if not collection_name.endswith("Session"):
            continue
        print("dealing with ", collection_name)
        if collection_name == "Wednesday-WorkingHours-Session":
            sample_rate = 0.2
        elif collection_name == "Friday-WorkingHours-Session":
            sample_rate = 0.5
        else:
            sample_rate = 10
        count = get_count(db, collection_name, thres=(8, 40))
        sample(db, collection_name, (8, 40), count, f_out, sample_rate)
    f_out.close()

        


















