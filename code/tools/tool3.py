import os
import random
import time

import pymongo
import tensorflow as tf
import ujson

from config.LabelMaps import label_maps
from tools.tool2 import pkt_to_pixel, reset_addr, str_to_pkt, bytes_encode

"""
    train valid test
"""


def _shuffle(arr):
    random.shuffle(arr)
    random.shuffle(arr)
    random.shuffle(arr)
    return arr


def final_data(oversample_dict, undersample_dict, new_db='mixed_613', raw_db='PacketInString'):
    """
        final_data(oversample_dict={'infiltration': 10,
                                    'bruteforce-web': 10,
                                    'bruteforce-xss': 20,
                                    'sql-injection': 40},
                   undersample_dict={'ddos-hoic': 4},
                   new_db='mixed_613',
                   raw_db='PacketInString')
    """
    train_ids, valid_ids, test_ids = [], [], []
    train_ids_us, train_ids_us_os = [], []

    client = pymongo.MongoClient()
    raw_db = client.get_database(raw_db)
    new_db = client.get_database(new_db)

    print(f'==> shuffling by cols...')
    col_names = raw_db.list_collection_names()
    for name in col_names:
        col = raw_db.get_collection(name)
        cur_ids = []
        for bs in col.find(no_cursor_timeout=True):
            cur_ids.append((name, bs['_id']))

        cur_ids = _shuffle(cur_ids)
        len1 = int(len(cur_ids) * 0.6)  # 613
        len2 = int(len(cur_ids) * 0.7)

        train_ids.extend(cur_ids[:len1])
        valid_ids.extend(cur_ids[len1:len2])
        test_ids.extend(cur_ids[len2:])

        label = name.lower()
        assert label in label_maps
        if label in oversample_dict:
            train_ids_us.extend(cur_ids[:len1])
            new_ids = []
            repeat_times = oversample_dict[label]
            for pair in cur_ids[:len1]:
                cnt = 0
                while cnt < repeat_times:
                    new_ids.append(pair)
                    cnt += 1
            train_ids_us_os.extend(new_ids)
        elif label in undersample_dict:
            len_ = len1 // undersample_dict[label]
            train_ids_us.extend(cur_ids[:len_])
            train_ids_us_os.extend(cur_ids[:len_])
        else:
            train_ids_us.extend(cur_ids[:len1])
            train_ids_us_os.extend(cur_ids[:len1])

    print(' ===== train ===== ')
    train_ids = _shuffle(train_ids)
    train_col = new_db.get_collection('train')
    _insert_by_id(train_col, train_ids, raw_db)

    print(' ===== train_undersample ===== ')
    train_ids_us = _shuffle(train_ids_us)
    train_us_col = new_db.get_collection('train_us')
    _insert_by_id(train_us_col, train_ids_us, raw_db)

    print(' ===== train_undersample_oversample ==== ')
    train_ids_us_os = _shuffle(train_ids_us_os)
    train_us_os_col = new_db.get_collection('train_us_os')
    _insert_by_id(train_us_os_col, train_ids_us_os, raw_db)

    print(' ===== validation ===== ')
    valid_ids = _shuffle(valid_ids)
    valid_col = new_db.get_collection('valid')
    _insert_by_id(valid_col, valid_ids, raw_db)

    print(' ===== test ===== ')
    test_ids = _shuffle(test_ids)
    test_col = new_db.get_collection('test')
    _insert_by_id(test_col, test_ids, raw_db)


def _insert_by_id(target_col, pairs, original_db):
    for pair in pairs:
        name, bid = pair
        bson_targets = original_db.get_collection(name).find({'_id': bid}, no_cursor_timeout=True)
        tmp = [x for x in bson_targets]
        assert len(tmp) == 1
        sample = tmp[0]
        sample.pop('_id')
        target_col.insert_one(sample)


"""
    mixed_str --> mixed_sparse --> sparse_tfrecord
"""


def _to_sparse(sparse_db, str_db, col_name):
    str_col = str_db.get_collection(col_name)
    sp_col = sparse_db.get_collection(col_name)
    for bson_ds in str_col.find(no_cursor_timeout=True):
        last_time = bson_ds['last_seen_time'] - bson_ds['begin_time']
        byte_len = bson_ds['byte_len']
        idx1, idx2, values = pkt_to_sparse_format(bson_ds['pkts_list'])
        label = label_maps[bson_ds['label']]
        sp_col.insert_one({
            'idx1': idx1,
            'idx2': idx2,
            'vals': values,
            'last_time': last_time,
            'byte_len': byte_len,
            'label': label
        })


def str_to_sparse_mongo(col_name, sparse_db_name='mixed_sparse', str_db_name='mixed_str'):
    assert col_name in ['train', 'valid', 'test'], 'Check collection name.'
    client = pymongo.MongoClient()
    sparse_db = client.get_database(sparse_db_name)
    str_db = client.get_database(str_db_name)
    _to_sparse(sparse_db, str_db, col_name)
    client.close()


def sparse_mongo_to_tfrecord(save_dir, col_name, sparse_db_name='mixed_sparse', bs=10000):
    save_dir = os.path.join(save_dir, col_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    assert col_name in ['train', 'valid', 'test']
    client = pymongo.MongoClient()
    db = client.get_database(sparse_db_name)
    col = db.get_collection(col_name)

    pa_cn = 0
    save_path = os.path.join(save_dir, 'part-%03d.tfrecord' % pa_cn)
    tfrecord_writer = tf.io.TFRecordWriter(save_path)

    sample_n = 0
    for bson_ds in col.find(no_cursor_timeout=True):
        example = serialize_sparse_example(
            bson_ds['idx1'],
            bson_ds['idx2'],
            bson_ds['vals'],
            bson_ds['label'],
            bson_ds['last_time'],
            bson_ds['byte_len']
        )
        tfrecord_writer.write(example)
        sample_n += 1

        if bs > 0 and sample_n % bs == 0:
            tfrecord_writer.close()
            pa_cn += 1
            save_path = os.path.join(save_dir, 'part-%03d.tfrecord' % pa_cn)
            tfrecord_writer = tf.io.TFRecordWriter(save_path)

    tfrecord_writer.close()
    client.close()


def str_to_sparse_mongo_to_tfrecord_main():
    str_to_sparse_mongo('train')
    str_to_sparse_mongo('test')
    str_to_sparse_mongo('valid')

    sparse_mongo_to_tfrecord(save_dir='/home/fgtc/Documents/sparse_mongo_tfrecord',
                             col_name='train', sparse_db_name='mixed_sparse', bs=10000)
    sparse_mongo_to_tfrecord(save_dir='/home/fgtc/Documents/sparse_mongo_tfrecord',
                             col_name='valid', sparse_db_name='sparse_mongo_tfrecord', bs=10000)
    sparse_mongo_to_tfrecord(save_dir='/home/fgtc/Documents/tfrecords',
                             col_name='test', sparse_db_name='sparse_mongo_tfrecord', bs=10000)


"""
    unused
"""


def oversample(labels_repeat_times_dict,
               db_name='mixed_str_613', col_name='train'):
    client = pymongo.MongoClient()
    assert db_name in client.list_database_names()
    db = client.get_database(db_name)
    assert col_name in db.list_collection_names()
    raw_col = db.get_collection(col_name)
    new_col = db.get_collection(col_name + '_oversample')
    tmp_col = db.get_collection('tmp')

    bids = []
    try:
        for sample in raw_col.find(no_cursor_timeout=True):
            label = sample['label']
            sample.pop('_id')
            res = tmp_col.insert_one(sample)
            bids.append(res.inserted_id)
            if label in labels_repeat_times_dict:
                cnt = 0
                repeat_times = labels_repeat_times_dict[label]
                while cnt < repeat_times:
                    sample.pop('_id')
                    res = tmp_col.insert_one(sample)
                    bids.append(res.inserted_id)
                    cnt += 1
        bids = _shuffle(bids)
        for bson_id in bids:
            sample = tmp_col.find({'_id': bson_id}, no_cursor_timeout=True)
            t = [i for i in sample]
            assert len(t) == 1
            sample = t[0]
            sample.pop('_id')
            new_col.insert_one(sample)
    except Exception as e:
        raise Exception(e)
    finally:
        db.drop_collection('tmp')
        client.close()


def shuffle_by_col_and_mixed(mixed_db_name='mixed_str_613', original_db_name='PacketInString'):
    train_ids, valid_ids, test_ids = [], [], []

    client = pymongo.MongoClient()
    original_db = client.get_database(original_db_name)
    mixed_db = client.get_database(mixed_db_name)

    print(f'==> shuffling by cols...')
    col_names = original_db.list_collection_names()
    for name in col_names:
        col_ids = []
        col = original_db.get_collection(name)
        for bs in col.find(no_cursor_timeout=True):
            col_ids.append((name, bs['_id']))

        col_ids = _shuffle(col_ids)
        # 采样1/4
        if name.lower() == 'ddos-hoic':
            print(f'==> Sampling {name.lower()}..')
            tmp_len = int(len(col_ids) // 4)
            col_ids = col_ids[:tmp_len]

        # 712
        if mixed_db_name.endswith('712'):
            len1 = int(len(col_ids) * 0.7)
            len2 = int(len(col_ids) * 0.8)
        # 613
        elif mixed_db_name.endswith('613'):
            len1 = int(len(col_ids) * 0.6)
            len2 = int(len(col_ids) * 0.7)
        else:
            raise AttributeError('Make sure the right ratio: 613 or 712..')

        train_ids.extend(col_ids[:len1])
        valid_ids.extend(col_ids[len1:len2])
        test_ids.extend(col_ids[len2:])

    print(f'==> mixing train...')
    train_ids = _shuffle(train_ids)
    train_col = mixed_db.get_collection('train')
    _insert_by_id(train_col, train_ids, original_db)

    print(f'==> mixing valid...')
    valid_ids = _shuffle(valid_ids)
    valid_col = mixed_db.get_collection('valid')
    _insert_by_id(valid_col, valid_ids, original_db)

    print(f'==> mixing test...')
    test_ids = _shuffle(test_ids)
    test_col = mixed_db.get_collection('test')
    _insert_by_id(test_col, test_ids, original_db)

    client.close()


def split_to_mongo(json_path, col_name):
    client = pymongo.MongoClient()
    train_db = client.get_database('train_split')
    valid_db = client.get_database('valid_split')
    test_db = client.get_database('test_split')

    train_col = train_db.get_collection(col_name)
    valid_col = valid_db.get_collection(col_name)
    test_col = test_db.get_collection(col_name)

    a, b, c = 0, 0, 0

    fr = open(json_path, 'r')
    for idx, line in enumerate(fr):
        bson = ujson.loads(line)
        bson.pop('_id')
        assert '_id' not in bson.keys()
        n = idx % 10
        if n < 7:
            res = train_col.insert_one(bson)
            a += 1
            assert res.inserted_id
        elif n < 8:
            res = valid_col.insert_one(bson)
            b += 1
            assert res.inserted_id
        else:
            res = test_col.insert_one(bson)
            c += 1
            assert res.inserted_id
    fr.close()
    client.close()
    print(f'train: {a}, valid: {b}, test: {c}')


def mix_and_shuffle():
    client = pymongo.MongoClient()
    mix_db = client.get_database('mixed_str')

    train_db = client.get_database('train_split')
    tuples = []
    for name in train_db.list_collection_names():
        col = train_db.get_collection(name)
        for son in col.find(no_cursor_timeout=True):
            bid = son['_id']
            tuples.append((name, bid))
    random.shuffle(tuples)
    random.shuffle(tuples)
    train_col = mix_db.get_collection('train')
    for pair in tuples:
        name_, bid_ = pair
        bson_targets = train_db.get_collection(name_).find({'_id': bid_}, no_cursor_timeout=True)
        tmp = [x for x in bson_targets]
        assert len(tmp) == 1
        target = tmp[0]
        target.pop('_id')
        train_col.insert_one(target)

    valid_db = client.get_database('valid_split')
    valid_col = mix_db.get_collection('valid')
    for name in valid_db.list_collection_names():
        tmp_col = valid_db.get_collection(name)
        for bson_ds in tmp_col.find(no_cursor_timeout=True):
            bson_ds.pop('_id')
            valid_col.insert_one(bson_ds)

    test_db = client.get_database('test_split')
    test_col = mix_db.get_collection('test')
    for name in test_db.list_collection_names():
        tmp_col = test_db.get_collection(name)
        for bson_ds in tmp_col.find(no_cursor_timeout=True):
            bson_ds.pop('_id')
            test_col.insert_one(bson_ds)

    client.close()


def split_main():
    s = time.time()
    ro = '/home/fgtc/Documents/mongo_export_str'
    files = os.listdir(ro)
    for p in files:
        print(p)
        split_to_mongo(os.path.join(ro, p), p[:-5])
    print(f'cost: {(time.time() - s) // 60} min')

    s = time.time()
    mix_and_shuffle()
    print(f'cost: {(time.time() - s) // 60} min')


def pixel_mongo_to_tfrecord(save_dir, col_name, pixel_db_name='data_pixel', bs=10000):
    save_dir = os.path.join(save_dir, col_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    assert col_name in ['train', 'val', 'test']
    client = pymongo.MongoClient()
    db = client.get_database(pixel_db_name)
    col = db.get_collection(col_name)

    pa_cn = 0
    save_path = os.path.join(save_dir, 'part-%03d.tfrecord' % pa_cn)
    tfrecord_writer = tf.io.TFRecordWriter(save_path)

    sample_n = 0
    for bson_ds in col.find(no_cursor_timeout=True):
        idx1, idx2, vals = [], [], []
        pkts = bson_ds['pkts_list']
        last_time = bson_ds['last_seen_time'] - bson_ds['begin_time']
        for i, p in enumerate(pkts):
            for j, ele in enumerate(p):
                if ele > 0:
                    idx1.append(i)
                    idx2.append(j)
                    vals.append(ele)
        label = label_maps[bson_ds['label']]
        example = serialize_sparse_example(
            idx1,
            idx2,
            vals,
            label,
            last_time,
            bson_ds['byte_len']
        )
        tfrecord_writer.write(example)
        sample_n += 1

        if bs > 0 and sample_n % bs == 0:
            tfrecord_writer.close()
            pa_cn += 1
            save_path = os.path.join(save_dir, 'part-%03d.tfrecord' % pa_cn)
            tfrecord_writer = tf.io.TFRecordWriter(save_path)

    tfrecord_writer.close()
    client.close()


def _img_help(bson, img_size=50 * 50, channels=100, mask=0):
    res = []
    for i, ele in enumerate(bson['pkts_list']):
        if i >= channels:
            break
        pixels = pkt_to_pixel(reset_addr(str_to_pkt(ele)))
        if len(pixels) < img_size:
            pixels.extend([mask] * (img_size - len(pixels)))
            res.extend(pixels)
        else:
            res.extend(pixels[:img_size])
    bl = len(bson['pkts_list'])
    while bl < channels:
        res.extend([mask] * img_size)
        bl += 1

    assert len(res) == img_size * channels
    return res


def _img_help_2(bson, img_size=50 * 50, channels=100, mask=0):
    res = []
    for i, ele in enumerate(bson['pkts_list']):
        if i >= channels:
            break
        pixels = pkt_to_pixel(reset_addr(str_to_pkt(ele)))
        if len(pixels) < img_size:
            res.append(pixels + [mask] * (img_size - len(pixels)))
        else:
            res.append(pixels[:img_size])

    assert len(res[-1]) == img_size
    assert len(res) <= channels
    return {
        'last_time': bson['begin_time'] - bson['last_seen_time'],
        'byte_len': bson['byte_len'],
        'features': res,
        'label': bson['label']
    }


def trans_img_half(name):
    client = pymongo.MongoClient()
    mix_db = client.get_database('mixed_str')
    half_img = client.get_database('half_mixed_img')

    assert name in mix_db.list_collection_names()
    print(f'===> Handle on {name}..')
    col = mix_db.get_collection(name)
    img_col = half_img.get_collection(name)
    for bs in col.find(no_cursor_timeout=True):
        img_col.insert_one(_img_help_2(bs))

    client.close()


def serialize_example(features, last_time, byte_len, label):
    assert isinstance(features, list)
    assert isinstance(last_time, float)
    assert isinstance(byte_len, int)
    assert isinstance(label, int)
    feature = {
        'pkts': tf.train.Feature(float_list=tf.train.FloatList(value=features)),
        'last_time': tf.train.Feature(float_list=tf.train.FloatList(value=[last_time])),
        'byte_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[byte_len])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),

    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_tfrecord(record_root='/home/fgtc/Documents/tfrecords'):
    client = pymongo.MongoClient()

    # train
    tfrecord_writer = tf.io.TFRecordWriter(os.path.join(record_root, 'train.tfrecord'))
    db = client.get_database('mixed_str')

    # valid
    # tfrecord_writer = tf.io.TFRecordWriter(os.path.join(record_root, 'valid.tfrecord'))
    # db = client.get_database('valid_split')

    # test
    # tfrecord_writer = tf.io.TFRecordWriter(os.path.join(record_root, 'test.tfrecord'))
    # db = client.get_database('test_split')

    for name in db.list_collection_names():
        col = db.get_collection(name)
        for bson_ds in col.find(no_cursor_timeout=True):
            if bson_ds['label'] == 'benign2':
                bson_ds['label'] = 'benign'
            assert bson_ds['label'] in label_maps
            label = label_maps[bson_ds['label']]
            last_time = bson_ds['last_seen_time'] - bson_ds['begin_time']
            byte_len = bson_ds['byte_len']
            fs = _img_help(bson_ds)
            tfrecord_writer.write(serialize_example(fs, last_time, byte_len, label))

    tfrecord_writer.close()
    client.close()


def _parse(exam_proto):
    feature_description = {
        'pkts': tf.io.FixedLenFeature([250000], tf.float32),
        'last_time': tf.io.FixedLenFeature([], tf.float32),
        'byte_len': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    ds = tf.io.parse_single_example(exam_proto, feature_description)
    features = ds['pkts']
    label = ds['label']
    return features, label


def read_tfrecord(path='./train.tfrecord'):
    ds = tf.data.TFRecordDataset([path], num_parallel_reads=tf.data.experimental.AUTOTUNE) \
        .map(_parse, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .batch(64).enumerate().unbatch()
    cnt = 0
    for _ in ds:
        if cnt > 100:
            break


""" 
    mixed_str --> sparse_tfrecord
"""


def serialize_sparse_example(idx1, idx2, vals, label, last_time, byte_len):
    assert len(idx1) == len(idx2) == len(vals)
    features = {
        'idx1': tf.train.Feature(int64_list=tf.train.Int64List(value=idx1)),
        'idx2': tf.train.Feature(int64_list=tf.train.Int64List(value=idx2)),
        'val': tf.train.Feature(int64_list=tf.train.Int64List(value=vals)),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'last_time': tf.train.Feature(float_list=tf.train.FloatList(value=[last_time])),
        'byte_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[byte_len]))
    }

    #     context = {
    #         'idx1': tf.train.Feature(int64_list=tf.train.Int64List(value=idx1)),
    #         'idx2': tf.train.Feature(int64_list=tf.train.Int64List(value=idx2)),
    #         'val': tf.train.Feature(int64_list=tf.train.Int64List(value=vals)),
    #     }
    #     feature_list = {
    #         'label': tf.train.FeatureList(
    #             feature=[
    #                 tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    #                 tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    #                 tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    #             ]
    #         )
    #     }
    #     example_proto = tf.train.SequenceExample(context=tf.train.Features(feature=context),
    #                                              feature_lists=tf.train.FeatureLists(feature_list=feature_list))

    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()


def pkt_to_sparse_format(pkts_list, img_size=50, channels=100):
    assert isinstance(pkts_list, list)
    idx1, idx2, vals = [], [], []
    for i, ele in enumerate(pkts_list):
        if i >= channels:
            break
        for j, b in enumerate(bytes_encode(reset_addr(str_to_pkt(ele)))):
            if j >= img_size * img_size:
                break
            pixel = int(b)
            if pixel > 0:
                idx1.append(i)
                idx2.append(j)
                vals.append(pixel)
    return idx1, idx2, vals


def parse_sparse_example(example_proto, channels=100, img_size=50):
    #     features = {
    #         'idx1': tf.io.VarLenFeature(dtype=tf.int64),
    #         'idx2': tf.io.VarLenFeature(dtype=tf.int64),
    #         'val': tf.io.VarLenFeature(dtype=tf.int64),
    #         'label': tf.io.FixedLenFeature(dtype=tf.int64)
    #     }

    features = {
        'sparse': tf.io.SparseFeature(index_key=['idx1', 'idx2'],
                                      value_key='val',
                                      dtype=tf.int64,
                                      size=[channels, img_size * img_size]),
        'label': tf.io.FixedLenFeature([], dtype=tf.int64),
        'byte_len': tf.io.FixedLenFeature([], dtype=tf.int64),
        'last_time': tf.io.FixedLenFeature([], dtype=tf.float64),
    }

    #     context = {
    #         'idx1': tf.io.VarLenFeature(dtype=tf.int64),
    #         'idx2': tf.io.VarLenFeature(dtype=tf.int64),
    #         'val': tf.io.VarLenFeature(dtype=tf.int64),
    #     }
    #     sequence_features = {
    #         'label': tf.io.VarLenFeature(dtype=tf.int64)
    #     }
    #     return tf.io.parse_sequence_example(example, context_features=context, sequence_features=sequence_features)

    return tf.io.parse_example(example_proto, features)


def str_mongo_to_sparse_tfrecord(save_dir, col_name, db_name='mixed_str', bs=10000):
    save_dir = os.path.join(save_dir, col_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    client = pymongo.MongoClient()
    db = client.get_database(db_name)
    col = db.get_collection(col_name)

    pa_cn = 0
    save_path = os.path.join(save_dir, 'part-%03d.tfrecord' % pa_cn)
    tfrecord_writer = tf.io.TFRecordWriter(save_path)

    sample_n = 0
    for bson_ds in col.find(no_cursor_timeout=True):

        label = label_maps[bson_ds['label']]
        last_time = bson_ds['last_seen_time'] - bson_ds['begin_time']
        byte_len = bson_ds['byte_len']
        idx1, idx2, values = pkt_to_sparse_format(bson_ds['pkts_list'])
        tfrecord_writer.write(serialize_sparse_example(idx1, idx2, values, label, last_time, byte_len))
        sample_n += 1

        if bs > 0 and sample_n % bs == 0:
            tfrecord_writer.close()
            pa_cn += 1
            save_path = os.path.join(save_dir, 'part-%03d.tfrecord' % pa_cn)
            tfrecord_writer = tf.io.TFRecordWriter(save_path)

    tfrecord_writer.close()
    client.close()


def str_mongo_to_sparse_tfrecord_main():
    str_mongo_to_sparse_tfrecord(save_dir='/home/fgtc/Documents/tfrecord',
                                 col_name='train', db_name='mixed_str', bs=10000)

    str_mongo_to_sparse_tfrecord(save_dir='/home/fgtc/Documents/tfrecord',
                                 col_name='test', db_name='mixed_str', bs=10000)

    str_mongo_to_sparse_tfrecord(save_dir='/home/fgtc/Documents/tfrecord',
                                 col_name='valid', db_name='mixed_str', bs=10000)
