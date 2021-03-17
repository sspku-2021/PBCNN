# -*- coding: utf-8 -*-

import sys
sys.path.append('..')


import config

import os
import time
import copy
import pickle
import json
import datetime


def time_trans(timestamp: str, hours: int = -3):
    timestamp = time.strptime(timestamp, "%d/%m/%Y %H:%M")
    timestamp = datetime.datetime.fromtimestamp(time.mktime(timestamp))
    timestamp = str(timestamp + datetime.timedelta(hours=hours))
    timestamp = time.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    timestamp = time.strftime("%-d/%-m/%-Y %-H:%-M", timestamp)
    return timestamp


class TimeParser():
    def __init__(self):
        self.flow_id_idx = 0
        self.timestamp_idx = 6
        self.label_idx = -1
        self.time_thres = self.get_time_thres()
        self.file_name_map = {"Friday-WorkingHours.pcap": ["Friday-WorkingHours-Morning.pcap_ISCX.csv",
                                                           "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
                                                           "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"],
                              "Thursday-WorkingHours.pcap": ["Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                                                             "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"],
                              "Wednesday-WorkingHours.pcap": ["Wednesday-workingHours.pcap_ISCX.csv"],
                              "Tuesday-WorkingHours.pcap": ["Tuesday-WorkingHours.pcap_ISCX.csv"],
                              "Monday-WorkingHours.pcap": ["Monday-WorkingHours.pcap_ISCX.csv"]}

    def get_time_flow_map(self, file_name):
        file_in = os.path.join(config.csv_dir, file_name)
        file_out = file_in[:-4] + "_time" + ".pkl"

        if os.path.exists(file_out):
            with open(file_out, "rb") as pkl:
                return pickle.load(pkl)

        attack2flow_id = dict()
        dirty_flow_id = set()

        is_afternoon = False
        if "Afternoon" in file_name:
            is_afternoon = True

        with open(file_in, "r", encoding="utf-8", errors="ignore") as fr:
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
                    timestamp = info[self.timestamp_idx]
                    if is_afternoon:
                        timestamp = time_trans(timestamp, 12)
                    flow_id += "-" + timestamp
                    attack2flow_id[label].add(flow_id)

        # dirty_flow_id:对于label为"BENIGN"的flow_id，如果该flow_id存在于attack2flow_id中，
        # 则该flow_id为dirty数据，加入dirty_flow_id集合中
        with open(file_in, "r", encoding="utf-8", errors="ignore") as fr:
            for idx, info in enumerate(fr):
                if idx == 0:
                    continue
                else:
                    info = info.rstrip().split(",")
                    label = info[self.label_idx]
                    flow_id = info[self.flow_id_idx] + "-" + info[self.timestamp_idx]
                    if flow_id in dirty_flow_id:
                        continue
                    if label == "BENIGN":
                        for _label in attack2flow_id:
                            if flow_id in attack2flow_id[_label]:
                                dirty_flow_id.add(flow_id)
                                attack2flow_id[_label].remove(flow_id)

        with open(file_out, "wb") as pkl:
            pickle.dump((attack2flow_id, dirty_flow_id), pkl)

        file_out = file_out[:-3] + "json"
        attack2flow_id_json, dirty_flow_id_json = copy.deepcopy(attack2flow_id), list(copy.deepcopy(dirty_flow_id))
        # attack2flow_id_json = copy.deepcopy(attack2flow_id)
        for label in attack2flow_id_json:
            attack2flow_id_json[label] = list(attack2flow_id_json[label])
        with open(file_out, "w", encoding="utf-8") as fr:
            json.dump((attack2flow_id_json, dirty_flow_id_json), fr, indent=4, ensure_ascii=False)

        return attack2flow_id, dirty_flow_id

    def get_time_thres(self):
        ret = dict()
        ret["Tuesday"] = {
            "FTP-Patator": (int(time.mktime(time.strptime("4/7/2017 9:20", "%d/%m/%Y %H:%M"))),
                            int(time.mktime(time.strptime("4/7/2017 10:20", "%d/%m/%Y %H:%M")))),
            "SSH-Patator": (int(time.mktime(time.strptime("4/7/2017 14:00", "%d/%m/%Y %H:%M"))),
                            int(time.mktime(time.strptime("4/7/2017 15:00", "%d/%m/%Y %H:%M"))))
        }
        ret["Wednesday"] = {
            "DoS slowloris": (int(time.mktime(time.strptime("5/7/2017 9:47", "%d/%m/%Y %H:%M"))),
                              int(time.mktime(time.strptime("5/7/2017 10:10", "%d/%m/%Y %H:%M")))),
            "DoS Slowhttptest": (int(time.mktime(time.strptime("5/7/2017 10:14", "%d/%m/%Y %H:%M"))),
                                 int(time.mktime(time.strptime("5/7/2017 10:35", "%d/%m/%Y %H:%M")))),
            "DoS Hulk": (int(time.mktime(time.strptime("5/7/2017 10:43", "%d/%m/%Y %H:%M"))),
                         int(time.mktime(time.strptime("5/7/2017 11:07", "%d/%m/%Y %H:%M")))),
            "DoS GoldenEye": (int(time.mktime(time.strptime("5/7/2017 11:10", "%d/%m/%Y %H:%M"))),
                              int(time.mktime(time.strptime("5/7/2017 11:23", "%d/%m/%Y %H:%M")))),
            "Heartbleed": (int(time.mktime(time.strptime("5/7/2017 15:12", "%d/%m/%Y %H:%M"))),
                           int(time.mktime(time.strptime("5/7/2017 15:32", "%d/%m/%Y %H:%M")))),
        }
        ret["Thursday"] = {
            "Web Attack  Brute Force": (int(time.mktime(time.strptime("6/7/2017 9:20", "%d/%m/%Y %H:%M"))),
                                        int(time.mktime(time.strptime("6/7/2017 10:00", "%d/%m/%Y %H:%M")))),
            "Web Attack  XSS": (int(time.mktime(time.strptime("6/7/2017 10:15", "%d/%m/%Y %H:%M"))),
                                int(time.mktime(time.strptime("6/7/2017 10:35", "%d/%m/%Y %H:%M")))),
            "Web Attack  Sql Injection": (int(time.mktime(time.strptime("6/7/2017 10:40", "%d/%m/%Y %H:%M"))),
                                          int(time.mktime(time.strptime("6/7/2017 10:42", "%d/%m/%Y %H:%M")))),
            "Infiltration": (int(time.mktime(time.strptime("6/7/2017 14:53", "%d/%m/%Y %H:%M"))),
                             int(time.mktime(time.strptime("6/7/2017 15:00", "%d/%m/%Y %H:%M")))),
        }
        ret["Friday"] = {
            "PortScan": (int(time.mktime(time.strptime("7/7/2017 13:55", "%d/%m/%Y %H:%M"))),
                         int(time.mktime(time.strptime("7/7/2017 15:23", "%d/%m/%Y %H:%M")))),
            "DDoS": (int(time.mktime(time.strptime("7/7/2017 15:56", "%d/%m/%Y %H:%M"))),
                     int(time.mktime(time.strptime("7/7/2017 16:16", "%d/%m/%Y %H:%M")))),
        }

        for key, infos in ret.items():
            for attack, t in infos.items():
                assert t[0] < t[1]

        return ret


class LabelMapTime():
    def __init__(self):
        self.label_pos = -1
        self.flow_id_pos = 1
        self.flow_id_swap_pos = 2
        self.timestamp_pos = 3
        self.file_name_map = {"Friday-WorkingHours_id2label.txt": ["Friday-WorkingHours-Morning.pcap_ISCX.csv",
                                                                   "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
                                                                   "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"],
                              "Thursday-WorkingHours_id2label.txt": ["Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                                                                     "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"],
                              "Wednesday-WorkingHours_id2label.txt": ["Wednesday-workingHours.pcap_ISCX.csv"],
                              "Tuesday-WorkingHours_id2label.txt": ["Tuesday-WorkingHours.pcap_ISCX.csv"],
                              "Monday-WorkingHours_id2label.txt": ["Monday-WorkingHours.pcap_ISCX.csv"]}

    def combine(self, attack2flow_id):
        _dic, _dirty = dict(), set()
        for attack_dic, _set in attack2flow_id:
            _dirty |= _set
            for attack, flow_id_set in attack_dic.items():
                if attack not in _dic:
                    _dic[attack] = set()
                _dic[attack] = _dic[attack] | flow_id_set
        return _dic, _dirty

    def id2name(self, parser, file_name, diff_hours=-3):
        file_in = os.path.join(config.cache_dir, file_name)
        file_out = file_in[:-4] + "_time_filter.txt"

        csv_file_in = [fn for fn in self.file_name_map[file_name]]
        _ = [parser.get_time_flow_map(_csv) for _csv in csv_file_in]
        attack2flow_id, dirty_set = self.combine(_)

        cnt, cnt_a, cnt_un = 0, 0, 0

        f_out = open(file_out, "w", encoding="utf-8")
        with open(file_in, "r", encoding="utf-8") as fr:
            for line in fr:
                cnt += 1
                line_list = line.rstrip().split("\t")
                if line[self.label_pos] != "BENIGN":
                    timestamp_diff = time_trans(line_list[self.timestamp_pos], diff_hours)
                    flow_id = line_list[1] + "-" + timestamp_diff
                    swap_flow_id = line_list[2] + "-" + timestamp_diff
                    if flow_id in dirty_set or swap_flow_id in dirty_set:
                        cnt_un += 1
                        f_out.write(line.rstrip() + "\t" + "DIRTY" + "\n")
                        # print("dirty data {} with attack {}".format(flow_id, line_list[-1]))
                    else:
                        is_attack = False
                        for attack, flow_id_set in attack2flow_id.items():
                            if flow_id in flow_id_set:
                                is_attack = True
                                f_out.write(line.rstrip() + "\t" + attack + "\n")
                                cnt_a += 1
                                break
                            elif swap_flow_id in flow_id_set:
                                is_attack = True
                                f_out.write(line.rstrip() + "\t" + attack + "\n")
                        if not is_attack:
                            f_out.write(line.rstrip() + "\t" + "BENIGN" + "\n")
                else:
                    f_out.write(line.rstrip() + "\t" + "BENIGN" + "\n")
        f_out.close()
        print("There are {} data with {} attack and {} uncertain.".format(cnt, cnt_a, cnt_un))


if __name__ == "__main__":
    label_map = LabelMapTime()
    time_parser = TimeParser()
    for file_name in os.listdir(config.cache_dir):
        if file_name.endswith("_id2label.txt"):
            label_map.id2name(time_parser, file_name)

