# -*- coding: utf-8 -*-
import sys
sys.path.append("../")

import config
import os
import pymongo
from tqdm import tqdm
from collections import Counter
from abc import ABCMeta, abstractmethod
import string
import socket
import struct
import gc
import random
from scapy.utils import hexdump
from PBCNN.tools.preprocess import str2bytes, bytes2str


class Session(metaclass=ABCMeta):
    def label_confirm(self, labels):
        counter = Counter(labels)
        label, count = counter.most_common(1)[0]
        if label == "BENIGN":
            if count != len(labels):
                return None
            else:
                return label
        if label == "uncertain":
            return None
        return label

    @abstractmethod
    def session_split(self, pkts):
        pass


class TCPSession(Session):
    def __init__(self):
        self.flags = {
            'F': 'FIN',
            'S': 'SYN',
            'R': 'RST',
            'P': 'PSH',
            'A': 'ACK',
            'U': 'URG',
            'E': 'ECE',
            'C': 'CWR',
        }

    def is_handshake(self, info):
        if info["flags"][0] != ["SYN"] or info["acks"][0] != 0:
            return False
        if info["flags"][1] != ['SYN', 'ACK'] or info["acks"][1] != info["seqs"][0] + 1:
            return False
        for i in range(2, len(info["bytes"])):
            if info["bytes"][i] == info["bytes"][i-1]:
                continue
            else:
                break
        if "ACK" not in info["flags"][i] or info["seqs"][i] != info["seqs"][0] + 1 or info["acks"][i] != info["seqs"][1] + 1:
            return False
        return True

    def is_closed(self, info):
        if "RST" not in info["flags"][-1]:
            return False
        return True

    def is_wave(self, info):
        idx = -1
        for i in range(len(info["bytes"]) - 4, 2, -1):
            if "FIN" in info["flags"][i]:
                idx = i
                break
        if idx == -1:
            return False
        if "ACK" not in info["flags"][idx + 1] or info["acks"][idx + 1] != info["seqs"][idx] + 1:
            return False
        if info["flags"][-2] != ['FIN', 'ACK'] or info["acks"][-2] != info["seqs"][idx] + 1:
            return False
        if info["flags"][-1] != ["ACK"] or info["seqs"][-1] != info["seqs"][idx] + 1 or info["acks"][-1] != info["seqs"][-2] + 1:
            return False
        return True

    def is_finish(self, info):
        return self.is_closed(info) or self.is_wave(info)

    def is_legal_tcp(self, info):
        return self.is_handshake(info) and self.is_finish(info)

    def session_split(self, pkts, log_out, thres=7):
        """

        """
        sessions = []
        session = dict()
        for idx, pkt in enumerate(pkts):
            tcp = pkt["bytes"].getlayer("TCP")
            F = [self.flags[x] for x in str(tcp.flags)]
            if idx == 0 or (F == ["SYN"] and tcp.ack == 0):
                if session.get("bytes") and len(session["bytes"]) > thres:
                    session["label"] = self.label_confirm(session["labels"])
                    if session["label"]:
                        if (session["label"] != "BENIGN" and session["labels"] != "DIRTY") or self.is_legal_tcp(session):
                            del session["acks"]
                            del session["seqs"]
                            del session["flags"]
                            del session["labels"]
                            sessions.append(session)
                        else:
                            if random.uniform(0, 100) < 1:
                                log_out.write("illegal tcp session with tuple 5: %s" % session["tuple_5"] + "\n")
                                log_out.write("session flags sequence: %s" % str(session["flags"]) + "\n")
                                log_out.write("session seq  sequence: %s" % " ".join(map(str, session["seqs"])) + "\n")
                                log_out.write("session ack sequence: %s" % " ".join(map(str, session["acks"])) + "\n")
                                log_out.write("" + "\n")
                    else:
                        if random.uniform(0, 10) < 1:
                            log_out.write("uncertain tcp label with tuple 5: %s" % session["tuple_5"] + "\n")
                            log_out.write("session label sequence: %s" % " ".join(session["labels"]) + "\n")
                            log_out.write("" + "\n")
                session = dict()
                session["id"] = pkt["id"]
                session["tuple_5"] = pkt["tuple_5"]
                session["bytes"] = []
                session["labels"] = []
                session["flags"] = []
                session["seqs"] = []
                session["acks"] = []
            session["bytes"].append(pkt["bytes"])
            session["labels"].append(pkt["label"])
            session["flags"].append(F)
            session["seqs"].append(tcp.seq)
            session["acks"].append(tcp.ack)

        if session.get("labels") and len(session["bytes"]) > thres:
            session["label"] = self.label_confirm(session["labels"])
            if session["label"]:
                if (session["label"] != "BENIGN" and session["labels"] != "DIRTY") or self.is_legal_tcp(session):
                    del session["acks"]
                    del session["seqs"]
                    del session["flags"]
                    del session["labels"]
                    sessions.append(session)
                else:
                    if random.uniform(0, 100) < 1:
                        log_out.write("illegal tcp session with tuple 5: %s" % session["tuple_5"] + "\n")
                        log_out.write("session flags sequence: %s" % str(session["flags"]) + "\n")
                        log_out.write("session seq  sequence: %s" % " ".join(map(str, session["seqs"])) + "\n")
                        log_out.write("session ack sequence: %s" % " ".join(map(str, session["acks"])) + "\n")
                        log_out.write("" + "\n")
            else:
                if random.uniform(0, 10) < 1:
                    log_out.write("uncertain tcp label with tuple 5: %s" % session["tuple_5"] + "\n")
                    log_out.write("session label sequence: %s" % " ".join(session["labels"]) + "\n")
                    log_out.write("" + "\n")

        return sessions


class UDPSession(Session):
    def session_split(self, pkts, log_out, thres=7, time_thres=90):
        sessions = []
        session = dict()
        session["bytes"] = []
        session["labels"] = []
        for idx, pkt in enumerate(pkts):
            if len(session["bytes"]) > 0:
                if pkt["bytes"].time - session["bytes"][-1].time >= time_thres:
                    if len(session["bytes"]) > thres:
                        session["label"] = self.label_confirm(session["labels"])
                        if session["label"]:
                            del session["labels"]
                            sessions.append(session)
                        else:
                            if random.uniform(0, 10) < 1:
                                log_out.write("uncertain tcp label with tuple 5: %s" % session["tuple_5"] + "\n")
                                log_out.write("session label sequence: %s" % " ".join(session["labels"]) + "\n")
                                log_out.write("" + "\n")
                    pkt["bytes"] = []
                    pkt["labels"] = []
            session["tuple_5"] = pkt["tuple_5"]
            session["id"] = pkt["id"]
            session["bytes"].append(pkt["bytes"])
            session["labels"].append(pkt["label"])
        if session.get("labels") and len(session["labels"]) > thres:
            session["label"] = self.label_confirm(session["labels"])
            if session["label"]:
                del session["labels"]
                sessions.append(session)
            else:
                if random.uniform(0, 10) < 1:
                    log_out.write("uncertain tcp label with tuple 5: %s" % session["tuple_5"] + "\n")
                    log_out.write("session label sequence: %s" % " ".join(session["labels"]) + "\n")
                    log_out.write("" + "\n")
        return sessions


class PostProcessor():
    def ip_mask(self, session):
        def get_random_ip():
            RANDOM_IP_POOL = ['192.168.10.222/0']
            str_ip = RANDOM_IP_POOL[random.randint(0, len(RANDOM_IP_POOL) - 1)]
            str_ip_addr = str_ip.split('/')[0]
            str_ip_mask = str_ip.split('/')[1]
            ip_addr = struct.unpack('>I', socket.inet_aton(str_ip_addr))[0]
            mask = 0x0
            for i in range(31, 31 - int(str_ip_mask), -1):
                mask = mask | (1 << i)
            ip_addr_min = ip_addr & (mask & 0xffffffff)
            ip_addr_max = ip_addr | (~mask & 0xffffffff)
            return socket.inet_ntoa(struct.pack('>I', random.randint(ip_addr_min, ip_addr_max)))

        mask_ip_src = get_random_ip()
        mask_ip_dst = get_random_ip()
        ip_src, ip_dst, _, _, _ = session["tuple_5"].split("-")
        ip_map = {ip_src: mask_ip_src,
                  ip_dst: mask_ip_dst}
        for i in range(len(session["bytes"])):
            session["bytes"][i].getlayer("IP").src = ip_map[session["bytes"][i].getlayer("IP").src]
            session["bytes"][i].getlayer("IP").dst = ip_map[session["bytes"][i].getlayer("IP").dst]
        return session

    def mac_mask(self, session):
        def get_random_mac():
            s = string.hexdigits
            mac = ":".join(["".join(random.sample(s, 2)[:2]).lower() for _ in range(6)])
            return mac

        mask_mac_src = get_random_mac()
        mask_mac_dst = get_random_mac()

        ether = session["bytes"][0].getlayer("Ether")
        mac_src, mac_dst = ether.src, ether.dst
        mac_map = {
            mac_src: mask_mac_src,
            mac_dst: mask_mac_dst
        }
        for i in range(len(session["bytes"])):
            session["bytes"][i].getlayer("Ether").src = mac_map[session["bytes"][i].getlayer("Ether").src]
            session["bytes"][i].getlayer("Ether").dst = mac_map[session["bytes"][i].getlayer("Ether").dst]
        return session

    def getHeadAndPayloads(self, session):
        def _get_head_payloads(pkt):
            pkt = hexdump(pkt, dump=True)
            head_info, payloads = [], []
            for line in pkt.split("\n"):
                line = line.strip().split("  ")
                head_info.append(line[1].lower().replace(" ", ""))
                if len(line) == 3:
                    payloads.append(line[2])
            return "".join(head_info), "".join(payloads)

        session["payloads"] = []
        session["bytes_head"] = []
        for i in range(len(session["bytes"])):
            head_info, payloads = _get_head_payloads(session["bytes"][i])
            session["bytes_head"].append(head_info)
            session["payloads"].append(payloads)
        session["payloads"] = "".join(session["payloads"])
        return session

    def get_flow_size(self, session):
        flow_size = sum([pkt.getlayer("IP").len for pkt in session["bytes"]])
        return flow_size

    def get_duration(self, session):
        return float(session["bytes"][-1].time - session["bytes"][0].time)


def split(collection_in,  collection_out, tcp_session: TCPSession, udp_session: UDPSession, post_processor: PostProcessor, log_out, thres=10000):
    # tuple_5_set_ = collection_in.aggregate([{"$group": {"_id": "$tuple_5"}}])
    # tuple_5_set_ = set(["".join(list(info.values())) for info in tuple_5_set_])
    """
    一个五元组对应多个数据包，针对每个五元组做处理，根据五元组的协议确定是TCP还是UDP；
    对处理后的数据包依次做ip_mask、mac_mask、getHeadAndPayloads、get_flow_size、get_duration；
    对每个数据包中的bytes，从字节格式转成字符串格式。。。
    最后将数据包依次插入到数据库中
    session格式：bytes、label、payloads_size、duration_time、tuple_5
    """
    tuple_5_set_ = collection_in.distinct("tuple_5")
    for idx, tuple_5 in enumerate(tqdm(tuple_5_set_)):
        infos = collection_in.find({"tuple_5": tuple_5}, {"_id": 0})
        print('type of infos:', type(infos))
        if infos.count() > thres:
            # ???
            continue
        infos = [info for info in infos]
        for idx, info in enumerate(infos):
            if idx == 0:
                protocol = int(info["tuple_5"].split("-")[-1])
            info["bytes"] = str2bytes(info["bytes"])
        assert protocol == 6 or protocol == 17
        if protocol == 6:
            sessions = tcp_session.session_split(infos, log_out)
        elif protocol == 17:
            sessions = udp_session.session_split(infos, log_out)
        for i in range(len(sessions)):
            if sessions[i]["label"] == "DIRTY":
                continue
            sessions[i] = post_processor.ip_mask(sessions[i])
            sessions[i] = post_processor.mac_mask(sessions[i])
            sessions[i] = post_processor.getHeadAndPayloads(sessions[i])
            sessions[i]["payloads_size"] = post_processor.get_flow_size(sessions[i])
            sessions[i]["duration_time"] = post_processor.get_duration(sessions[i])
            for j in range(len(sessions[i]["bytes"])):
                sessions[i]["bytes"][j] = bytes2str(sessions[i]["bytes"][j])
        for session in sessions:
            try:
                if session["label"] == "DIRTY":
                    continue
                collection_out.insert(session)
            except:
                print("too large session with %d" % len(session["bytes"]))
        if idx % 100 == 0:
            gc.collect()


if __name__ == "__main__":
    host = 'mongodb://localhost:27017/'
    db_name = 'IDS_2017'
    client = pymongo.MongoClient(host)
    db = client[db_name]
    tcp_session = TCPSession()
    udp_session = UDPSession()
    post_processor = PostProcessor()
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    collection_names = [collection_name for collection_name in db.list_collection_names() if collection_name.endswith("WorkingHours")]
    for collection_name in collection_names:
        if collection_name not in ["Friday-WorkingHours", "Thursday-WorkingHours"]:
            continue
        print(f"processing {collection_name}")
        collection_out_name = collection_name + "-Session"
        if collection_out_name in db.list_collection_names():
            print(f"dropping {collection_out_name}")
            db.drop_collection(collection_out_name)
        collection_in = db[collection_name]
        collection_out = db[collection_name + "-Session"]
        collection_out.create_index("label")
        log_path = os.path.join(config.log_dir, collection_name)
        log_out = open(log_path, "w", encoding="utf-8")
        split(collection_in, collection_out, tcp_session, udp_session, post_processor, log_out)
        log_out.close()








