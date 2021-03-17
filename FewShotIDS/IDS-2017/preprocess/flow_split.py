# -*- coding: utf-8 -*-

import sys
sys.path.append("../")

import config
import os
import shutil
import pymongo
import json
from tqdm import tqdm
from scapy.utils import PcapReader
from scapy import layers
from scapy.all import bytes_hex


class five_tuple_distribute:
    def __init__(self):
        self.database = self.create_database("IDS_2017")

    def id2attack(self, filename):
        id2attack = dict()
        with open(filename, "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.rstrip().split("\t")
                id, label = int(line[0]), line[-1]
                id2attack[id] = label
        return id2attack

    def create_database(self, dbname, host='mongodb://localhost:27017/', is_new_db=False):
        # create a new database
        client = pymongo.MongoClient(host)
        if is_new_db and dbname in client.list_database_names():
            client.drop_database(dbname)
        db = client[dbname]
        return db

    def __extract_5_tuple(self, pkt):
        # get the 5_tuple to the packet
        if layers.inet.IP in pkt.layers():
            ip = pkt.getlayer("IP")
            ip_src = ip.src
            ip_dst = ip.dst
            protocol = ip.proto
            if protocol == 0:
                print("Getting unknown protocol {}.".format(protocol))
                return None
            elif protocol == 6:
                tcp = pkt.getlayer("TCP")
                try:
                    assert tcp is not None
                except:
                    print("Getting unknown tcp data.")
                    return None
                sport = tcp.sport
                dport = tcp.dport
            elif protocol == 17:
                udp = pkt.getlayer("UDP")
                try:
                    assert udp is not None
                except:
                    print("Getting unknown udp data.")
                    return None
                sport = udp.sport
                dport = udp.sport
            else:
                print("Getting unexpected protocol that not occur in TrafficLabelling.")
                return None
            return "{}-{}-{}-{}-{}".format(ip_src, ip_dst, sport, dport, protocol)
        else:
            return None

    def __call__(self, filename, iswrite_disk=False, thres=10):
        pcap_filename = os.path.join(config.pcap_dir, filename)
        label_filename = os.path.join(config.cache_dir, filename.split(".")[0] + "_id2label_time_filter.txt")
        id2attack = self.id2attack(label_filename)
        dir_name = os.path.join(config.cache_dir, filename.split(".")[0])
        collection_name = filename.split(".")[0]
        print(collection_name)

        # according to five tuples, the data distribution is stored in mongodb
        print("Storing data in mongodb")
        if collection_name in self.database.list_collection_names():
            self.database[collection_name].drop()
        collection = self.database[collection_name]
        collection.create_index("tuple_5")
        tuple_5_set = set()
        print(pcap_filename)
        with PcapReader(pcap_filename) as pcap_reader:
            for idx, pkt in enumerate(pcap_reader):
                if idx % 100000 == 0:
                    print("processing {} data".format(idx))
                tuple_5 = self.__extract_5_tuple(pkt)
                if tuple_5:
                    ip_src, ip_dst, sport, dport, protocol = tuple_5.split("-")
                    bytes_str = bytes.decode(bytes_hex(pkt))
                    label = id2attack.get(idx, "uncertain")
                    if label == "uncertain":
                        print("getting uncertain label with idx {}".format(idx))
                    tuple_5_swap = "{}-{}-{}-{}-{}".format(ip_dst, ip_src, dport, sport, protocol)
                    # if tuple_5_swap exists, save tuple_5_swap, otherwise save tuple_5
                    if tuple_5_swap in tuple_5_set:
                        info = {"id": idx,
                                "bytes": bytes_str,
                                "tuple_5": tuple_5_swap,
                                "label": label}
                        tuple_5_set.add(tuple_5_swap)
                    else:
                        info = {"id": idx,
                                "bytes": bytes_str,
                                "tuple_5": tuple_5,
                                "label": label}
                        tuple_5_set.add(tuple_5)
                    # five tuples, one direction
                    assert not ((tuple_5 in tuple_5_set) and (tuple_5_swap in tuple_5_set))
                    collection.insert(info)

        # tuple_5_set_ = collection.distinct("tuple_5")
        # distinct too big, 16mb cap
        tuple_5_set_ = collection.aggregate([{"$group": {"_id": "$tuple_5"}}])
        tuple_5_set_ = set(["".join(list(info.values())) for info in tuple_5_set_])
        assert tuple_5_set == tuple_5_set_  # enhanced robustness

        # delete short tuple_5
        for tuple_5 in tqdm(tuple_5_set_):
            if collection.find({"tuple_5": tuple_5}).count() < thres:
                collection.delete_many({"tuple_5": tuple_5})

        # local disk as backup
        if iswrite_disk:
            print("Writing data to disk...")
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
            os.mkdir(dir_name)
            for tuple_5 in tqdm(tuple_5_set_):
                with open(os.path.join(dir_name, tuple_5), "w", encoding="utf-8") as f_out:
                    infos = collection.find({"tuple_5": tuple_5}, {"_id": 0})
                    for info in infos:
                        info = json.dumps(info)
                        f_out.write(info + "\n")


if __name__ == "__main__":
    distributer = five_tuple_distribute()
    for file_name in os.listdir(config.pcap_dir):
        if file_name.endswith(".pcap"):
            distributer(file_name)

