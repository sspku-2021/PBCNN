# -*- coding: utf-8 -*-
import sys
sys.path.append("../")

import argparse
import json
import os
import datetime
import logging

import utils
import config

from model.siamese_network import Siamese
from model import train, test


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/Val/Test various models on the IDS")
    parser.add_argument(
        "--start_epoch", type=int, default=-1,
        help="which epoch to restore from (-1 means starting from scratch)")
    parser.add_argument(
        "--num_epochs", type=int, default=20,
        help="number of epochs to train")
    parser.add_argument(
        "--batch_size", help="batch size",
        type=int, default=128)
    parser.add_argument(
        "--mode", help="decide is to train or decode",
        type=str, default="train")
    parser.add_argument(
        "--is_debug", help="for debug",
        type=utils.str2bool, nargs='?', const=True, default=False)
    parser.add_argument(
        "--is_early_stop", help="does use val data for early stop",
        type=utils.str2bool, nargs='?', const=True, default=True
    )
    parser.add_argument(
        "--is_tuning", help="does use pretrain-model to tuning",
        type=utils.str2bool, nargs="?", const=True, default=False
    )
    parser.add_argument(
        "--train_file_path", help="train file path",
        default=None, type=str,
    )
    parser.add_argument(
        "--dev_file_path", help="dev file path",
        default=None, type=str,
    )
    parser.add_argument(
        "--test_file_path", help="test file path",
        default=None, type=str,
    )
    parser.add_argument(
        "--params_path", help="parameters' path",
        default=None, type=str,
    )
    parser.add_argument(
        "--ckpt_dir", help="",
        default=None, type=str,
    )
    parser.add_argument(
        "--restore_ckpt", help="pretrain model ckpt",
        default=None, type=str,
    )
    parser.add_argument(
        "--file_output", help="output to store results",
        default=None, type=str,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    with open(args.params_path, "r", encoding="utf-8") as fr:
        hyperParams = json.load(fr)

    hyperParams.update({
        "is_train": True if args.mode == "train" else False,
        "early_stop": args.is_early_stop,
        "is_tuning": args.is_tuning,
    })
    model = Siamese(
        session_length=hyperParams["session_length"],
        height=hyperParams["height"],
        width=hyperParams["width"],
        learning_rate=hyperParams["learning_rate"],
        filter_sizes=hyperParams["filter_sizes"],
        num_filters=hyperParams["num_filters"],
        filter_sizes_hierarchical=hyperParams["filter_sizes_hierarchical"],
        num_fitlers_hierarchical=hyperParams["num_fitlers_hierarchical"],
        is_train=hyperParams["is_train"],
        is_tuning=hyperParams["is_tuning"]
    )

    if args.mode == "train":
        log_path = os.path.join(config.log_dir, f"siamese.{datetime.datetime.now().strftime('%m-%d-%H-%M')}.logs")
        utils.init_logging(log_path)
        logging.info(hyperParams)

        train.train_siamese(model, args.batch_size, args.num_epochs,
                    args.train_file_path, args.dev_file_path,
                     args.is_tuning, ckpt_dir=args.ckpt_dir,
                    restore_ckpt=args.restore_ckpt, start_epoch = args.start_epoch)

