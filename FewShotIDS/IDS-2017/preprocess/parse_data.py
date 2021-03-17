# -*- coding: utf-8 -*-
import sys
sys.path.append("../")
import config
import scapy
from scapy.all import *
from scapy.utils import PcapReader
from scapy import layers
import os
import datetime
import time
import pickle
import copy
import json
# from collections import defaultdict


class LabelMap():
    def __init__(self):
        self.csv_parser = CsvParser()
        self.pcap_parser = PcapParser()
        self.file_name_map = {"Friday-WorkingHours.pcap": ["Friday-WorkingHours-Morning.pcap_ISCX.csv",
                                                           "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
                                                           "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"],
                              "Thursday-WorkingHours.pcap": ["Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                                                             "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"],
                              "Wednesday-WorkingHours.pcap": ["Wednesday-workingHours.pcap_ISCX.csv"],
                              "Tuesday-WorkingHours.pcap": ["Tuesday-WorkingHours.pcap_ISCX.csv"],
                              "Monday-WorkingHours.pcap": ["Monday-WorkingHours.pcap_ISCX.csv"]}

    def combine(self, attact2flow_id):
        """
        merge items with the same key
        """
        dic = dict()
        for attact_dic in attact2flow_id:
            for attact, flow_id_set in attact_dic.items():
                if attact not in dic:
                    dic[attact] = set()
                dic[attact] = dic[attact] | flow_id_set
        return dic

    def id2label(self, file_name):
        pcap_file_in = os.path.join(config.pcap_dir, file_name)
        csv_file_in = [os.path.join(config.csv_dir, fn) for fn in self.file_name_map[file_name]]
        file_out = os.path.join(config.cache_dir, file_name.split(".")[0] + "_id2label.txt")

        attact2flow_id = [self.csv_parser.get_flow_map(csv_fi) for csv_fi in csv_file_in]
        attact2flow_id = self.combine(attact2flow_id)

        f_out = open(file_out, "w", encoding="utf-8")

        cnt_a, cnt = 0, 0

        with PcapReader(pcap_file_in) as pcap_reader:
            for idx, pkt in enumerate(pcap_reader):
                tuple_5 = self.pcap_parser._extract_5_tuple(pkt, idx)
                if tuple_5:
                    cnt += 1
                    is_attack = False
                    for attact, flow_id_set in attact2flow_id.items():
                        if tuple_5["5_tuple"] in flow_id_set:
                            # dic[attact] = dic.setdefault(attact, 0) + 1
                            f_out.write(str(tuple_5['id']) + "\t" + tuple_5["5_tuple"] + "\t" + tuple_5["5_tuple_swap"] + "\t" + tuple_5["timestamp"] + "\t" + attact + "\n")
                            is_attack = True
                            # cnt_a += 1
                            continue
                        elif tuple_5["5_tuple_swap"] in flow_id_set:
                            # dic[attact] = dic.setdefault(attact, 0) + 1
                            f_out.write(str(tuple_5['id']) + "\t" + tuple_5["5_tuple"] + "\t" + tuple_5["5_tuple_swap"] + "\t" + tuple_5["timestamp"] + "\t" + attact + "\n")
                            is_attack = True
                            # cnt_a += 1
                            continue
                    if not is_attack:
                        # dic['BENIGN'] = dic.setdefault("BENIGN", 0) + 1
                        f_out.write(str(tuple_5['id']) + "\t" + tuple_5["5_tuple"] + "\t" + tuple_5["5_tuple_swap"] + "\t" + tuple_5["timestamp"] + "\t" + "BENIGN" + "\n")

        # f_out.close()
        print("There are {} abnormal data with proportion {}".format(cnt_a, cnt_a/cnt))


class CsvParser():
    def __init__(self):
        self.flow_id_idx = 0
        self.timestamp_idx = 6
        self.label_idx = -1

    def get_flow_map(self, file_name):
        """
        仅仅抽了五元组， 时间策略和肮数据过滤之后再写， 这样速度快些
        将attack2flow_id序列化成pkl和json文件
        return: 一个dict，attack - flow_id pairs
        Getting attack flow id: "tuple-timestamp" and dirty flow id
        """
        file_out = file_name[:-3] + "pkl"
        if os.path.exists(file_out):
            with open(file_out, "rb") as pkl:
                return pickle.load(pkl)

        attack2flow_id = dict()
        print('CsvParser.get_flow_map()\'s param file_name:', file_name)

        with open(file_name, "r", encoding="utf-8", errors="ignore") as fr:
            for idx, info in enumerate(fr):
                if idx == 0:
                    continue
                else:
                    info = info.rstrip().split(",")
                    label = info[self.label_idx]
                    if label == "BENIGN":
                        continue
                    if label not in attack2flow_id:
                        attack2flow_id[label] = set()
                    flow_id = info[self.flow_id_idx]
                    attack2flow_id[label].add(flow_id)

        with open(file_out, "wb") as pkl:
            pickle.dump(attack2flow_id, pkl)

        file_out = file_name[:-3] + "json"
        attack2flow_id_json = copy.deepcopy(attack2flow_id)
        for label in attack2flow_id_json:
            attack2flow_id_json[label] = list(attack2flow_id_json[label])
        with open(file_out, "w", encoding="utf-8") as fr:
            json.dump(attack2flow_id_json, fr, indent=4, ensure_ascii=False)
        return attack2flow_id


class PcapParser():

    def _extract_5_tuple(self, pkt, idx):
        if layers.inet.IP in pkt.layers():
            timestamp = str(datetime.datetime.utcfromtimestamp(pkt.time))[:-7]
            try:
                timestamp = time.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                timestamp = time.strftime("%-d/%-m/%-Y %-H:%-M", timestamp)
            except:
                print("The data's format({}) is illegall".format(timestamp))
                return None
            ip = pkt.getlayer("IP")

            ip_src, ip_dst, protocol = ip.src, ip.dst, ip.proto

            if protocol == 0:
                print("Getting unknown protocol {}.".format(protocol))
                return None
            elif protocol == 6:
                tcp = pkt.getlayer("TCP")
                try:
                    assert tcp != None
                except:
                    print("Getting unknown tcp data.")
                    return None
                sport = tcp.sport
                dport = tcp.dport
            elif protocol == 17:
                udp = pkt.getlayer("UDP")
                try:
                    assert udp != None
                except:
                    print("Getting unknown udp data.")
                    return None
                sport = udp.sport
                dport = udp.sport
            else:
                print("Getting unexpected protocol that not occur in TrafficLabelling.")
                return None
            return {"5_tuple": "{}-{}-{}-{}-{}".format(ip_src, ip_dst, sport, dport, protocol),
                    "5_tuple_swap": "{}-{}-{}-{}-{}".format(ip_dst, ip_src, dport, sport, protocol),
                    "timestamp": timestamp,
                    "id": idx}

    def extract_5_tuple(self, file_name):
        tuple_5_list = []
        with PcapReader(file_name) as pcap_reader:
            for idx, pkt in enumerate(pcap_reader):
                tuple_5 = self._extract_5_tuple(pkt, idx)
                if tuple_5:
                    tuple_5_list.append(tuple_5)
        return tuple_5_list


if __name__ == "__main__":
    file_name_list = [file_name for file_name in os.listdir(config.pcap_dir) if file_name.endswith(".pcap")]
    label_map = LabelMap()
    # dic = defaultdict(int)
    for file_name in file_name_list:
        print("processing ", file_name)
        label_map.id2label(file_name)
        # print(dic)
    # print(dic)






