from tools.tool2 import save_benign_mongo
import time

import os
import sys
import time

from tools import tool2
from tools.tool3 import shuffle_by_col_and_mixed, str_mongo_to_sparse_tfrecord, oversample, final_data

def pcap_to_str_mongo():
	print("#### attack pcap ---> mongo ####")
	print("processing...")
    files = ['./data_cache/raw_attacker_packets/bruteforce-xss_22-02.pcap',
             './data_cache/raw_attacker_packets/dos-goldeneye_15-02.pcap',
             './data_cache/raw_attacker_packets/sql-injection_22-02.pcap',
             './data_cache/raw_attacker_packets/bruteforce-xss_23-02.pcap',
             './data_cache/raw_attacker_packets/ftp-bruteforce_14-02.pcap',
             './data_cache/raw_attacker_packets/dos-slowloris_15-02.pcap',
             './data_cache/raw_attacker_packets/infiltration_28-02.pcap',
             './data_cache/raw_attacker_packets/sql-injection_23-02.pcap',
             './data_cache/raw_attacker_packets/bruteforce-web_22-02.pcap',
             './data_cache/raw_attacker_packets/bot_02-03.pcap',
             './data_cache/raw_attacker_packets/dos-slowhttptest_16-02.pcap',
             './data_cache/raw_attacker_packets/bruteforce-web_23-02.pcap']
    for fi in files:
        tool2.save_str_mongo(pcap_file=fi,
                             database_name='PacketInString')

    dirs = [
        './data_cache/raw_attacker_packets/dos-hulk_16-02',
        './data_cache/raw_attacker_packets/ssh-bruteforce_14-02',
        './data_cache/raw_attacker_packets/ddos-loic-http_20-02',
        './data_cache/raw_attacker_packets/ddos-hoic_21-02'
        './data_cache/raw_attacker_packets/ddos-loic-udp_20-02',
        './data_cache/raw_attacker_packets/ddos-loic-udp_21-02'
    ]

    for di in dirs:
        tool2.save_str_mongo(pcap_file=di,
                             database_name='PacketInString')
    print("all attack done.")



def main():
	print("begin preprocessing...")
	# process benign pcap files and save into mogodb
    save_benign_mongo(pcap_dir='/home/fgtc/Documents/notebooks/data_cache/raw_benign_packets',
                      label='benign',
                      database_name='PacketInString')

    save_benign_mongo(pcap_dir='/home/fgtc/Documents/notebooks/data_cache/raw_benign_packets2',
                      label='benign2',
                      database_name='PacketInString')
    # process attack pcap files and save into Mongodb
    pcap_to_str_mongo()

    # split train, valid, test and save to Mongodb
    final_data(oversample_dict={'infiltration': 10,
                                'bruteforce-web': 10,
                                'bruteforce-xss': 20,
                                'sql-injection': 40},
               undersample_dict={'ddos-hoic': 4},
               new_db='mixed_613',
               raw_db='PacketInString')
    # convert train, valid, test to tfrecord.
    str_mongo_to_sparse_tfrecord(save_dir='/home/fgtc/Documents/tfrecord',
                                 col_name='train', db_name='mixed_str', bs=10000)

    str_mongo_to_sparse_tfrecord(save_dir='/home/fgtc/Documents/tfrecord',
                                 col_name='test', db_name='mixed_str', bs=10000)

    str_mongo_to_sparse_tfrecord(save_dir='/home/fgtc/Documents/tfrecord',
                                 col_name='valid', db_name='mixed_str', bs=10000)

    print("preprocess done!!!")


if __name__ == '__main__':
    s = time.time()
    main()
    print(f'cost: {(time.time() - s) // 60} min')

