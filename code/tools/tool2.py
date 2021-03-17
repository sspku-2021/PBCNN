# encoding = utf-8
import os

import pymongo
import ujson
from scapy.all import PcapReader
from scapy.compat import bytes_encode, hex_bytes, bytes_hex
from scapy.layers import l2

from config import Attacker_ips

"""
    Unused
"""


def get_flow_id(pkt):
    """
        根据packet获取五元组key
    """
    ip_layer = pkt.getlayer('IP')
    src_ip = ip_layer.src
    dst_ip = ip_layer.dst

    app_layer = ip_layer.payload
    sport = app_layer.sport
    dport = app_layer.dport

    pre = f'{src_ip}-{dst_ip}-{sport}-{dport}'
    proc = app_layer.name

    return f'{pre}-{proc}'


def wt_to_json(file, data_list):
    fw = open(file, 'w')
    json_data = '\n'.join(map(ujson.dumps, data_list))

    fw.write(json_data)
    fw.write('\n')
    fw.close()


def save_str_json(pcap_path, timeout=64, save_dir='./data_cahce/json_str_packet'):
    """
        割双向流
        [
            {
                'biflow_id': ,
                'pkts_list': [],
                'begin_time': pkt.time,
                'byte_len':
                'end': false
            },
        ]
    """

    if os.path.isdir(pcap_path):
        label_time_info = pcap_path.split(os.sep)[-1]
        pcap_path = [os.path.join(pcap_path, x) for x in sorted(os.listdir(pcap_path))]
    else:
        label_time_info = pcap_path.split(os.sep)[-1].split('.')[0]
        pcap_path = [pcap_path]
    label = label_time_info.split('_')[0].strip()

    flows_maps = dict()
    cnt = 0
    pro_wr = 0

    save_path = os.path.join(save_dir, f'{label_time_info}.json')
    print(f'===> Saving to {save_path}')
    fw = open(save_path, 'w')
    for path in pcap_path:
        with PcapReader(path) as pr:
            for pkt in pr:
                # if inet.IP not in pkt.layers():
                if not pkt.haslayer('IP'):
                    continue
                bid = get_biflow_id(pkt)
                if isinstance(bid, Exception):
                    pro_wr += 1
                    continue
                if cnt % 100 == 0:
                    fw.flush()
                biflow_id, protocol = bid
                pkt_str = pkt_to_str(pkt)

                if biflow_id in flows_maps:
                    cur_biflow = flows_maps[biflow_id]
                    last_seen_time = cur_biflow['last_seen_time']

                    if int(float(pkt.time) - last_seen_time) >= timeout:  # 间隔时间
                        cur_biflow['is_finished'] = True
                        cur_biflow['byte_len'] = sum(len(x) for x in cur_biflow['pkts_list']) // 2
                        fw.write(ujson.dumps(cur_biflow))
                        fw.write('\n')
                        cnt += 1
                        flows_maps[biflow_id] = {
                            'biflow_id': biflow_id,
                            'pkts_list': [pkt_str],
                            'begin_time': float(pkt.time),
                            'last_seen_time': float(pkt.time),
                            'is_finished': False,
                            'label': label
                        }
                    else:
                        cur_biflow['pkts_list'].append(pkt_str)
                        cur_biflow['last_seen_time'] = float(pkt.time)
                else:
                    flows_maps[biflow_id] = {
                        'biflow_id': biflow_id,
                        'pkts_list': [pkt_str],
                        'begin_time': float(pkt.time),
                        'last_seen_time': float(pkt.time),
                        'is_finished': False,
                        'label': label
                    }

    for ele in list(flows_maps.values()):
        ele['byte_len'] = sum(len(x) for x in ele['pkts_list']) // 2
        fw.write(ujson.dumps(ele))
        fw.write('\n')
        cnt += 1
    fw.flush()
    fw.close()
    print(f"Biflow cnt: {cnt} , wrong protocol: {pro_wr}, DONE !!!")


def _too_large_help_2(bson, limit=20000):
    bl = bson['byte_len']
    if bl < limit:
        return bson
    while bl >= limit:
        l = len(bson['pkts_list']) - 1
        bson['pkts_list'] = bson['pkts_list'][:l]
        bl = sum([len(x) for x in bson['pkts_list']]) // 2
    bson['is_cut'] = True
    return bson


def _img_help_2(biflow_json, img_size, mask=255):
    pkts_list = biflow_json['pkts_list']
    res = []
    for pkt in pkts_list:
        res += pkt
        if len(res) >= img_size:
            break
    if len(res) < img_size:
        res += [mask] * (img_size - len(res))
    else:
        res = res[:img_size]
    assert len(res) == img_size
    return {
        'biflow_id': biflow_json['biflow_id'],
        'begin_time': biflow_json['begin_time'],
        'pixel_pkts': res,
        'label': biflow_json['label']
    }


def save_pixel_mongo(pcap_file,
                     img_size=1600,
                     reset=False,
                     database_name='PacketInPixel',
                     timeout=64,
                     timeout2=120):
    if os.path.isdir(pcap_file):
        label_time = pcap_file.split(os.sep)[-1]
        pcap_paths = [os.path.join(pcap_file, x) for x in sorted(os.listdir(pcap_file))]
    else:
        label_time = pcap_file.split(os.sep)[-1][:-5]
        pcap_paths = [pcap_file]

    label, day_mon = label_time.split('_')
    day, mon = day_mon.split('-')
    t_ips = getattr(Attacker_ips, f'attacker_ips_{day}_{mon}')

    collection_name = label
    mongo_session = pymongo.MongoClient()
    mongo_db = mongo_session.get_database(database_name)
    if reset and collection_name in mongo_db.list_collection_names():
        mongo_db.drop_collection(collection_name)
    mongo_col = mongo_db.get_collection(collection_name)

    flows_maps = dict()
    cnt = 0
    pro_wr = 0
    for path in pcap_paths:
        print(f'===> Handle on  {path} ')
        with PcapReader(path) as pr:
            for pkt in pr:
                # if inet.IP not in pkt.layers():
                if not pkt.haslayer('IP'):
                    continue
                ip_layer = pkt.getlayer("IP")
                if ip_layer.src not in t_ips and ip_layer.dst not in t_ips:
                    continue
                bid = get_biflow_id(pkt)
                if isinstance(bid, Exception):
                    pro_wr += 1
                    continue

                biflow_id, protocol = bid
                pkt_pixel = pkt_to_pixel(reset_addr(pkt))

                if biflow_id in flows_maps:
                    cur_biflow = flows_maps[biflow_id]
                    last_seen_time = cur_biflow['last_seen_time']

                    if int(float(pkt.time) - cur_biflow['begin_time']) >= timeout2 \
                            or int(float(pkt.time) - last_seen_time) >= timeout:  # 持续时间 or 间隔时间
                        cur_biflow['is_finished'] = True
                        mongo_col.insert_one(_img_help_2(cur_biflow, img_size))
                        cnt += 1
                        flows_maps[biflow_id] = {
                            'biflow_id': biflow_id,
                            'pkts_list': [pkt_pixel],
                            'begin_time': float(pkt.time),
                            'last_seen_time': float(pkt.time),
                            'is_finished': False,
                            'label': label
                        }
                    else:
                        cur_biflow['pkts_list'].append(pkt_pixel)
                        cur_biflow['last_seen_time'] = float(pkt.time)
                else:
                    flows_maps[biflow_id] = {
                        'biflow_id': biflow_id,
                        'pkts_list': [pkt_pixel],
                        'begin_time': float(pkt.time),
                        'last_seen_time': float(pkt.time),
                        'is_finished': False,
                        'label': label
                    }

    for ele in list(flows_maps.values()):
        mongo_col.insert_one(_img_help_2(ele, img_size))
        cnt += 1
    print(f"Biflow cnt: {cnt} , wrong protocol: {pro_wr}, DONE !!!")
    mongo_session.close()


def img_shape(bson, img_size=50 * 50, channels=100, mask=0):
    res = []
    for i, ele in enumerate(bson['pkts_list']):
        if i >= channels:
            break
        pixel = pkt_to_pixel(reset_addr(str_to_pkt(ele)))
        if len(pixel) < img_size:
            res.append(pixel + [mask] * (img_size - len(pixel)))
        else:
            res.append(pixel[:img_size])
    bl = len(bson['pkts_list'])
    if bl < channels:
        res += [[mask] * img_size] * (channels - bl)

    assert len(res[-1]) == img_size
    assert len(res) == channels

    return {
        'biflow_id': bson['biflow_id'],
        'begin_time': bson['begin_time'],
        'end_time': bson['last_seen_time'],
        'byte_len': bson['byte_len'],
        'pixel_pkts': [','.join(map(str, x)) for x in res],
        'label': bson['label']
    }


def trans_pixel_by_col(col, pixel_db_name='PacketInPixel', str_db_name='PacketInString'):
    mongo_session = pymongo.MongoClient()
    str_db = mongo_session.get_database(str_db_name)
    pix_db = mongo_session.get_database(pixel_db_name)

    col_names = str_db.list_collection_names()
    assert col in col_names
    print(f'===> Handle on {col}')
    str_col = str_db.get_collection(col)
    pix_col = pix_db.get_collection(col)
    for ele in str_col.find(no_cursor_timeout=True):
        pix_col.insert_one(img_shape(ele))
    mongo_session.close()


def trans_pixel(pixel_db_name='PacketInPixel', str_db_name='PacketInString'):
    mongo_session = pymongo.MongoClient()

    str_db = mongo_session.get_database(str_db_name)
    col_names = str_db.list_collection_names()

    pix_db = mongo_session.get_database(pixel_db_name)
    for col in col_names:
        print(f'===> Handle on {col}')
        str_col = str_db.get_collection(col)
        tmp = []
        pix_col = pix_db.get_collection(col)
        for ele in str_col.find():
            tmp.append(img_shape(ele))
            # pix_col.insert_one(_img_shape(ele))
            if len(tmp) >= 100000:
                print(f'writing...')
                pix_col.insert_many(tmp)
                tmp.clear()
        pix_col.insert_many(tmp)

    mongo_session.close()


"""
    Used
"""


def get_biflow_id(pkt):
    """
        根据packet获取五元组key
    """
    assert pkt.haslayer('IP')
    ip_layer = pkt.getlayer('IP')
    src_ip = ip_layer.src
    dst_ip = ip_layer.dst

    app_layer = ip_layer.payload
    proc = app_layer.name

    try:
        sport = app_layer.sport
        dport = app_layer.dport

        if src_ip < dst_ip:
            pre = f'{src_ip}-{sport}-{dst_ip}-{dport}'
        else:
            pre = f'{dst_ip}-{dport}-{src_ip}-{sport}'

        return f'{pre}-{proc}', proc

    except Exception as e:
        return e


def reset_addr(pkt):
    """
        匿名化MAC地址和IP地址
    """
    assert pkt.haslayer('Ether') and pkt.haslayer('IP')
    pkt.getlayer('Ether').dst = "00:00:00:00:00:00"
    pkt.getlayer('Ether').src = "00:00:00:00:00:00"
    pkt.getlayer('IP').src = "0.0.0.0"
    pkt.getlayer('IP').dst = "0.0.0.0"
    return pkt


def pkt_to_pixel(pkt):
    """
        将字节转成像素值
    """
    return [int(b) for b in bytes_encode(pkt)]


def pkt_to_str(pkt):
    return bytes_hex(pkt).decode()


def str_to_pkt(string):
    return l2.Ether(hex_bytes(string))


def _too_large_help(bson, min_limit=1000000, max_limit=5000000):
    bl = bson['byte_len']
    if bl < max_limit:
        return bson
    cut_len = len(bson['pkts_list']) // 2
    cnt = 0
    while True:
        cnt += 1
        bl = sum([len(x) for x in bson['pkts_list'][:cut_len]]) // 2
        if bl < min_limit:
            cut_len = cut_len * 3 // 2
        elif min_limit <= bl < max_limit:
            break
        else:
            cut_len = cut_len // 2
        if cnt > 10000:
            raise RecursionError(f"==> byte_len: {bson['byte_len']}")
    bson['is_cut'] = True
    bson['pkts_list'] = bson['pkts_list'][:cut_len]
    return bson


def save_str_mongo(pcap_file,
                   database_name='PacketInString',
                   reset=False,
                   timeout=64,
                   timeout2=120,
                   file_buffer=20):
    if os.path.isdir(pcap_file):
        label_time = pcap_file.split(os.sep)[-1]
        pcap_paths = [os.path.join(pcap_file, x) for x in sorted(os.listdir(pcap_file))]
    else:
        label_time = pcap_file.split(os.sep)[-1][:-5]
        pcap_paths = [pcap_file]

    label, day_mon = label_time.split('_')
    day, mon = day_mon.split('-')
    a_ips = getattr(Attacker_ips, f'attacker_ips_{day}_{mon}')

    collection_name = label
    mongo_session = pymongo.MongoClient()
    mongo_db = mongo_session.get_database(database_name)
    if reset and collection_name in mongo_db.list_collection_names():
        mongo_db.drop_collection(collection_name)
    mongo_col = mongo_db.get_collection(collection_name)

    flows_maps = dict()
    cnt = 0
    pro_wr = 0
    for i, path in enumerate(pcap_paths):
        if i > 0 and i % file_buffer == 0:
            tmp = []
            for f in list(flows_maps.values()):
                f['byte_len'] = sum([len(x) for x in f['pkts_list']]) // 2
                cnt += 1
                tmp.append(_too_large_help(f))
            print(f'Writing {len(tmp)} to mongo .')
            mongo_col.insert_many(tmp)
            flows_maps.clear()
            del tmp
        print(f'===> Handle on {path} ')
        with PcapReader(path) as pr:
            for pkt in pr:
                # if inet.IP not in pkt.layers():
                if not pkt.haslayer('IP'):
                    continue
                ip_layer = pkt.getlayer("IP")
                if ip_layer.src not in a_ips and ip_layer.dst not in a_ips:
                    continue
                bid = get_biflow_id(pkt)
                if isinstance(bid, Exception):
                    pro_wr += 1
                    continue

                biflow_id, protocol = bid
                pkt_str = pkt_to_str(pkt)

                if biflow_id in flows_maps:
                    cur_biflow = flows_maps[biflow_id]
                    last_seen_time = cur_biflow['last_seen_time']

                    if int(float(pkt.time) - cur_biflow['begin_time']) >= timeout2 \
                            or int(float(pkt.time) - last_seen_time) >= timeout:  # 持续时间 or 间隔时间
                        cur_biflow['is_finished'] = True
                        cur_biflow['byte_len'] = sum([len(x) for x in cur_biflow['pkts_list']]) // 2
                        mongo_col.insert_one(_too_large_help(cur_biflow))
                        cnt += 1
                        flows_maps[biflow_id] = {
                            'biflow_id': biflow_id,
                            'pkts_list': [pkt_str],
                            'begin_time': float(pkt.time),
                            'last_seen_time': float(pkt.time),
                            'is_finished': False,
                            'label': label
                        }
                    else:
                        cur_biflow['pkts_list'].append(pkt_str)
                        cur_biflow['last_seen_time'] = float(pkt.time)
                else:
                    flows_maps[biflow_id] = {
                        'biflow_id': biflow_id,
                        'pkts_list': [pkt_str],
                        'begin_time': float(pkt.time),
                        'last_seen_time': float(pkt.time),
                        'is_finished': False,
                        'label': label
                    }

    for ele in list(flows_maps.values()):
        ele['byte_len'] = sum([len(x) for x in ele['pkts_list']]) // 2
        mongo_col.insert_one(_too_large_help(ele))
        cnt += 1
    print(f"Biflow cnt: {cnt} , wrong protocol: {pro_wr}, DONE !!!")
    mongo_session.close()


def save_benign_mongo(pcap_dir,
                      label='benign',
                      database_name='PacketInString',
                      reset=False,
                      timeout=64,
                      timeout2=120):
    assert os.path.isdir(pcap_dir)
    pcap_paths = [os.path.join(pcap_dir, x) for x in sorted(os.listdir(pcap_dir))]

    collection_name = label
    mongo_session = pymongo.MongoClient()
    mongo_db = mongo_session.get_database(database_name)
    if reset and collection_name in mongo_db.list_collection_names():
        mongo_db.drop_collection(collection_name)
    mongo_col = mongo_db.get_collection(collection_name)

    flow_cnt = 0
    tcp_pkt = 0
    udp_pkt = 0
    icmp_pkt = 0
    other_pkt = 0

    for path in pcap_paths:
        day_mon, ip = path.split(os.sep)[-1].split('_')
        day, mon = day_mon.split('-')
        assert ip not in getattr(Attacker_ips, f'attacker_ips_{day.strip()}_{mon.strip()}')

        flows_maps = dict()
        print(f'===> Handle on  {path} ')
        with PcapReader(path) as pr:
            for pkt in pr:
                if not pkt.haslayer('IP'):
                    continue
                bid = get_biflow_id(pkt)
                if isinstance(bid, Exception):
                    continue

                biflow_id, protocol = bid
                pkt_str = pkt_to_str(pkt)

                if protocol.lower() == 'tcp':
                    tcp_pkt += 1
                elif protocol.lower() == 'udp':
                    udp_pkt += 1
                elif protocol.lower() == 'icmp':
                    icmp_pkt += 1
                else:
                    other_pkt += 1

                if biflow_id in flows_maps:
                    cur_biflow = flows_maps[biflow_id]
                    last_seen_time = cur_biflow['last_seen_time']

                    # if int(float(pkt.time) - last_seen_time) >= timeout:
                    if int(float(pkt.time) - cur_biflow['begin_time']) >= timeout2 \
                            or int(float(pkt.time) - last_seen_time) >= timeout:  # 持续时间 or 间隔时间
                        cur_biflow['is_finished'] = True
                        cur_biflow['byte_len'] = sum([len(x) for x in cur_biflow['pkts_list']]) // 2
                        mongo_col.insert_one(_too_large_help(cur_biflow))
                        flow_cnt += 1
                        flows_maps[biflow_id] = {
                            'biflow_id': biflow_id,
                            'pkts_list': [pkt_str],
                            'begin_time': float(pkt.time),
                            'last_seen_time': float(pkt.time),
                            'is_finished': False,
                            'label': label
                        }
                    else:
                        cur_biflow['pkts_list'].append(pkt_str)
                        cur_biflow['last_seen_time'] = float(pkt.time)
                else:
                    flows_maps[biflow_id] = {
                        'biflow_id': biflow_id,
                        'pkts_list': [pkt_str],
                        'begin_time': float(pkt.time),
                        'last_seen_time': float(pkt.time),
                        'is_finished': False,
                        'label': label
                    }

        for ele in list(flows_maps.values()):
            ele['byte_len'] = sum([len(x) for x in ele['pkts_list']]) // 2
            mongo_col.insert_one(_too_large_help(ele))
            flow_cnt += 1
        del flows_maps

    mongo_session.close()
    print(f'tcp_pkt: {tcp_pkt}')
    print(f'udp_pkt: {udp_pkt}')
    print(f'icmp_pkt: {icmp_pkt}')
    print(f'other_pkt: {other_pkt}')
    print(f"Biflow cnt: {flow_cnt} , DONE !!!")
