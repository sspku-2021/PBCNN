# -*- coding: utf-8 -*-

from model.utils import modules
from model.cnn_model import CnnModel
from model.hierarchical_model import HierarchicalModel
from model.hierarchical_fusion_model import HierarchicalFusionModel


import logging
import argparse
import tensorflow as tf


def init_logging(file_name):
    logging.basicConfig(
        level = logging.INFO,
        format='%(asctime)s-%(levelname)s-%(message)s',
        datefmt='%y-%m-%d %H:%M',
        filename=file_name,
        filemode="w"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2func(v):
    return getattr(modules, v)


def params_update(hyperParams, args):
    if not args.is_pretrain:
        assert (not (args.is_size or args.is_time)) and (not args.uncertainty)
    else:
        if args.uncertainty:
            assert args.is_size and args.is_time
        else:
            assert args.is_size or args.is_time

    is_train = True if args.mode == "train" else False
    hyperParams.update({
        "is_train": is_train,
        "early_stop": args.is_early_stop,
        "is_pretrain": args.is_pretrain,
        "is_time": args.is_time,
        "is_size": args.is_size,
        "uncertainty": args.uncertainty,
        "is_tuning": args.is_tuning,
    })
    return hyperParams


def build_model(hyperParams, model_name):
    if model_name == "cnn":
        return CnnModel(session_length=hyperParams["session_length"],
                        height=hyperParams["height"],
                        width=hyperParams["width"],
                        learning_rate=hyperParams["learning_rate"],
                        num_labels=hyperParams["num_labels"],
                        num_output=hyperParams["num_output"],
                        is_pretrain=hyperParams["is_pretrain"],
                        is_time=hyperParams["is_time"],
                        is_size=hyperParams["is_size"],
                        is_train=hyperParams["is_train"],
                        uncertainty=hyperParams["uncertainty"],
                        is_tuning=hyperParams["is_tuning"]
                        )

    elif model_name == "hierarchical_cnn":
        return HierarchicalModel(
                        session_length=hyperParams["session_length"],
                        height=hyperParams["height"],
                        width=hyperParams["width"],
                        learning_rate=hyperParams["learning_rate"],
                        num_labels=hyperParams["num_labels"],
                        filter_sizes=hyperParams["filter_sizes"],
                        num_filters=hyperParams["num_filters"],
                        filter_sizes_hierarchical=hyperParams["filter_sizes_hierarchical"],
                        num_fitlers_hierarchical=hyperParams["num_fitlers_hierarchical"],
                        is_pretrain=hyperParams["is_pretrain"],
                        is_time=hyperParams["is_time"],
                        is_size=hyperParams["is_size"],
                        is_train=hyperParams["is_train"],
                        uncertainty=hyperParams["uncertainty"],
                        is_tuning=hyperParams["is_tuning"]
            )

    elif model_name == "cnn_lstm":
        return HierarchicalFusionModel(
            session_length=hyperParams["session_length"],
            height=hyperParams["height"],
            width=hyperParams["width"],
            learning_rate=hyperParams["learning_rate"],
            num_labels=hyperParams["num_labels"],
            filter_sizes=hyperParams["filter_sizes"],
            num_filters=hyperParams["num_filters"],
            lstm_size=hyperParams["lstm_size"],
            lstm_num_layers=hyperParams["lstm_num_layers"],
            lstm_dropout_rate=hyperParams["lstm_dropout_rate"],
            attention_size=hyperParams["attention_size"],
            average=hyperParams["average"],
            is_pretrain=hyperParams["is_pretrain"],
            is_time=hyperParams["is_time"],
            is_size=hyperParams["is_size"],
            is_train=hyperParams["is_train"],
            uncertainty=hyperParams["uncertainty"],
            is_tuning=hyperParams["is_tuning"]
        )
    else:
        raise NotImplementedError('select model in [cnn, hierarchical_cnn, cnn_lstm]')


def gpu_config():
    """
    Speicify configurations of GPU
    """
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'

    upper = 0.8
    config.gpu_options.per_process_gpu_memory_fraction = upper
    print("GPU memory upper bound:", upper)
    return config

