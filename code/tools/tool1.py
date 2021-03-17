# encoding = utf-8
import os
import sys
import time
from multiprocessing import Pool

from scapy.all import PcapReader, wrpcap

"""
    1. 获取时间分布，确定粗略时间段
    2. 根据粗略时间段获取浮点数时间段
    3. 按时间段过滤后存储
"""

"""
    Unused
"""


def _save_task(pcap_path, time_range, a_ips):
    time_range = sorted(time_range, key=lambda x: x.split('-')[0])
    ft_ranges = get_ftime(time_range, pcap_path, a_ips)
    pkt_list = []
    with PcapReader(pcap_path) as pr:
        j = 0
        for pkt in pr:
            ti = pkt.time
            if ti < ft_ranges[j][0]:
                continue
            elif ti <= ft_ranges[j][1]:
                pkt_list.append(pkt)
            else:
                j += 1
                if j >= len(time_range):
                    break
    return pkt_list, j


def save_pkts_2_(time_ranges, label, pcap_paths, a_ips, time_prefix,
                 save_dir='./data_cache/raw_attacker_packets', parallel_num=4):
    assert len(time_ranges) == len(pcap_paths)
    pools = Pool(parallel_num)
    p_objs = []
    counts = 0
    for i, p in enumerate(pcap_paths):
        p = pools.apply_async(_save_task, args=(p, time_ranges[i], a_ips,))
        p_objs.append(p)
    pools.close()
    pools.join()

    all_pkts = []
    for r in p_objs:
        pkts, n = r.get()
        counts += n
        all_pkts += pkts

    wt_path = os.path.join(save_dir, f'{label}_{time_prefix}.pcap')
    print(f'===> Packets cnt {len(all_pkts)}, writing to {wt_path}')
    wrpcap(wt_path, all_pkts)


def time_in_range(time_ranges, t):
    for tr in time_ranges:
        if tr[0] <= t <= tr[1]:
            return True
    return False


"""
    Used
"""


def _gf_single(pcap, begin_time_dic, end_time_dic, t_ips):
    print(f'Pcap path {pcap}, pid {os.getpid()} ..')
    assert len(begin_time_dic) == len(end_time_dic)

    begin_ex = [sys.maxsize] * len(begin_time_dic)
    end_ex = [0] * len(begin_time_dic)
    with PcapReader(pcap) as pr:
        for pkt in pr:
            ti = local_time(pkt.time)
            # if inet.IP in pkt.layers():
            if pkt.haslayer('IP'):
                ip_layer = pkt.getlayer('IP')
                if ip_layer.src in t_ips or ip_layer.dst in t_ips:
                    if ti in begin_time_dic:
                        idx = begin_time_dic[ti]
                        begin_ex[idx] = min(begin_ex[idx], pkt.time)
                    elif ti in end_time_dic:
                        idx = end_time_dic[ti]
                        end_ex[idx] = max(pkt.time, end_ex[idx])

    return begin_ex, end_ex


def _gf_mp(begin_dic, end_dic, p_paths, a_ips, process_num=4):
    pools = Pool(process_num)
    p_objs = []
    for path in p_paths:
        p = pools.apply_async(func=_gf_single, args=(path, begin_dic, end_dic, a_ips,))
        p_objs.append(p)
    pools.close()
    pools.join()

    begin_exs = [x.get()[0] for x in p_objs]
    end_exs = [x.get()[1] for x in p_objs]

    be, en = [], []
    for i in range(len(begin_dic)):
        be.append(min([x[i] for x in begin_exs]))
        en.append(max([x[i] for x in end_exs]))

    return [[be[i], en[i]] for i in range(len(be))]


def local_time(f_time):
    """
        根据浮点数时间返回本地时间
    """
    lc = time.localtime(f_time)
    ti = '%02d/%02d/%02d:%02d' % (lc.tm_mon, lc.tm_mday, lc.tm_hour, lc.tm_min)
    return ti


def _dis_task(pcap_path, t_ips):
    print(f'HANDLE on pcap path: {pcap_path}')
    tmp_dis = set()
    cnt = 0
    with PcapReader(pcap_path) as pr:
        for pkt in pr:
            # if inet.IP in pkt.layers():
            if pkt.haslayer('IP'):
                ip_layer = pkt.getlayer('IP')
                if ip_layer.src in t_ips or ip_layer.dst in t_ips:
                    cnt += 1
                    ti = local_time(pkt.time)
                    if ti not in tmp_dis:
                        tmp_dis.add(ti)
    return tmp_dis, cnt


def get_dis(p_paths, a_ips, parallel_num=4):
    """
        返回时间分布和攻击ip通信的数量
    """
    assert isinstance(p_paths, list)

    pools = Pool(parallel_num)
    res = []
    for p in p_paths:
        r = pools.apply_async(_dis_task, args=(p, a_ips,))
        res.append(r)
    pools.close()
    pools.join()

    all_dis = []
    target_cnts = 0
    for r in res:
        dis, n = r.get()
        target_cnts += n
        all_dis.append(sorted(list(dis)))

    return all_dis, target_cnts


def print_all_dis(all_dis):
    max_len = 0
    for ele in all_dis:
        if max_len < len(ele):
            max_len = len(ele)
    for i in range(max_len):
        tmp = []
        for j in range(len(all_dis)):
            if i >= len(all_dis[j]):
                tmp.append(" " * 11)
            else:
                tmp.append(all_dis[j][i])
        print(tmp)


def get_ftime(t_ranges, p_path, a_ips, process_num=4):
    """
            :param t_ranges: ['02/22/22:13-02/22/23:23',
                              '02/23/01:51-02/23/02:29',
                              '02/23/04:14-02/23/04:27']
            :param p_path: 路径或者目录
            :param a_ips: {'18.218.115.60'}
            :param process_num:
            :return: [[Decimal('1519308824.965705'), Decimal('1519313039.858533')],
                      [Decimal('1519321899.783923'), Decimal('1519324181.827037')],
                      [Decimal('1519330470.169342'), Decimal('1519331276.022793')]]
    """
    t_ranges = sorted(t_ranges, key=lambda x: x.split('-')[0])

    begin_dic = dict()
    for i, r in enumerate(t_ranges):
        begin_dic[r.split('-')[0]] = i

    end_dic = dict()
    for i, r in enumerate(t_ranges):
        end_dic[r.split('-')[1]] = i

    if os.path.isdir(p_path):
        p_paths = [os.path.join(p_path, x) for x in sorted(os.listdir(p_path)) if not x.startswith('.')]
        return _gf_mp(begin_dic, end_dic, p_paths, a_ips, process_num)
    else:
        begin_ex, end_ex = _gf_single(p_path, begin_dic, end_dic, a_ips)
        return [[begin_ex[i], end_ex[i]] for i in range(len(begin_ex))]


# 先遍历得到浮点数时间，然后按浮点数时间筛选
def save_pkts(time_labels, pcap_path, a_ips, time_prefix,
              save_dir='./data_cache/raw_attacker_packets', buffer_size=0):
    t_ranges = list(time_labels.keys())
    labels = list(time_labels.values())
    print("getting float times...")
    ft_ranges = get_ftime(t_ranges, pcap_path, a_ips)
    for i, ele in enumerate(ft_ranges):
        ele.append(labels[i])
    ft_ranges = sorted(ft_ranges, key=lambda x: x[0])
    # ft_ranges = [[Decimal('1518790334.134225'), Decimal('1518793513.888214'), 'DoS-SlowHTTPTest'],
    #              [Decimal('1518803127.826666'), Decimal('1518803902.127974'), 'DoS-Hulk']]
    # ft_ranges = [[Decimal('1518631281.199541'), Decimal('1518636750.726589'), 'SSH-BruteForce']]
    # ft_ranges = [[Decimal('1519222131.251323'), Decimal('1519224219.269411'), 'DDOS-LOIC-UDP']]

    print(ft_ranges)

    i = 0
    j = 0
    pkt_list = []
    if buffer_size > 0:
        # tmp_dir = os.path.join(save_dir, f'{time_prefix}_{ft_ranges[i][2]}')
        tmp_dir = os.path.join(save_dir, f'{ft_ranges[i][2].lower()}_{time_prefix}')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

    if os.path.isdir(pcap_path):
        pcap_path = [os.path.join(pcap_path, x) for x in sorted(os.listdir(pcap_path))]
    else:
        pcap_path = [pcap_path]

    for pcap in pcap_path:
        print(f'Handle on pcap path {pcap} ...')
        with PcapReader(pcap) as pr:
            for pkt in pr:
                if buffer_size > 0 and len(pkt_list) > 0 and len(pkt_list) % buffer_size == 0:
                    tmp_path = os.path.join(tmp_dir, 'part-%03d' % j)
                    print(f'===> Packets cnt {len(pkt_list)}, Writing to {tmp_path}')
                    wrpcap(tmp_path, pkt_list)  # 尝试新建进程存储节约时间，失败。

                    # del pkt_list
                    # pkt_list = []
                    pkt_list.clear()
                    j += 1
                ti = pkt.time
                if ti < ft_ranges[i][0]:
                    continue
                elif ti <= ft_ranges[i][1]:
                    pkt_list.append(pkt)
                else:
                    assert len(pkt_list) > 0
                    if buffer_size > 0 and j > 0:
                        wt_path = os.path.join(tmp_dir, 'part-%03d' % j)
                    else:
                        wt_path = os.path.join(save_dir, f'{ft_ranges[i][2].lower()}_{time_prefix}.pcap')

                    print(f'===> Packets cnt {len(pkt_list)}, writing to {wt_path}')
                    wrpcap(wt_path, pkt_list)
                    # del pkt_list
                    # pkt_list = []
                    pkt_list.clear()
                    i += 1
                    j = 0
                    if i >= len(ft_ranges):
                        break
                    if buffer_size > 0:
                        tmp_dir = os.path.join(save_dir, f'{ft_ranges[i][2].lower()}_{time_prefix}')
                        if not os.path.exists(tmp_dir):
                            os.makedirs(tmp_dir)


# 先按local_time字符串时间筛选存储，然后按浮点数时间精确筛选
def save_pkts_3(time_labels, pcap_path, a_ips, time_prefix,
                save_dir='./data_cache/raw_attacker_packets', buffer_size=0):
    time_labels_ = list(zip(time_labels.keys(), time_labels.values()))
    time_labels_ = sorted(time_labels_, key=lambda x: x[0])

    t_ranges = [x[0] for x in time_labels_]
    labels = [x[1] for x in time_labels_]

    all_pkts = []
    i = 0
    j = 0
    if buffer_size > 0:
        tmp_dir = os.path.join(save_dir, f'tmp/{time_prefix}')
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

    with PcapReader(pcap_path) as pr:
        tmp_times = t_ranges[i].split('-')
        for pkt in pr:
            if buffer_size > 0 and len(all_pkts) > 0 and len(all_pkts) % buffer_size == 0:
                tmp_path = os.path.join(tmp_dir, 'part-%03d' % j)
                print(f'===> Packets cnt {len(all_pkts)}, Writing to {tmp_path}')
                wrpcap(tmp_path, all_pkts)
                # del all_pkts
                # all_pkts = []
                all_pkts.clear()
                j += 1
            ti = local_time(pkt.time)
            if ti < tmp_times[0]:
                continue
            elif ti <= tmp_times[1]:
                all_pkts.append(pkt)
            else:
                i += 1
                if i >= len(t_ranges):
                    break
                tmp_times = t_ranges[i].split('-')

    if buffer_size > 0 and j > 0:
        tmp_path = os.path.join(tmp_dir, 'part-%03d' % j)
        print(f'===> Packets cnt {len(all_pkts)}, Writing to {tmp_path}')

        wrpcap(tmp_path, all_pkts)
        del all_pkts
        # all_pkts.clear()
    save_pkts(time_labels, tmp_dir, a_ips, time_prefix, save_dir=save_dir, buffer_size=buffer_size)


# 合并label相同但是时间段不同的数据
def save_pkts_2(time_ranges, label, pcap_paths, a_ips, time_prefix,
                save_dir='./data_cache/raw_attacker_packets'):
    assert len(time_ranges) == len(pcap_paths)
    all_pkts = []
    for i, p in enumerate(pcap_paths):
        with PcapReader(p) as pr:
            j = 0
            tmp_ranges = time_ranges[i]  # list
            ft_ranges = get_ftime(tmp_ranges, p, a_ips)
            print(ft_ranges)
            for pkt in pr:
                ti = pkt.time
                if ti < ft_ranges[j][0]:
                    continue
                elif ti <= ft_ranges[j][1]:
                    all_pkts.append(pkt)
                else:
                    j += 1
                    if j >= len(tmp_ranges):
                        break

    wt_path = os.path.join(save_dir, f'{label}_{time_prefix}.pcap')
    print(f'===> Packets cnt {len(all_pkts)}, writing to {wt_path}')
    wrpcap(wt_path, all_pkts)
