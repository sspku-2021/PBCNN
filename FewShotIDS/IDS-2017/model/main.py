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
        "--model_name", help="model type",
        type=str, default="cnn")
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
        "--is_pretrain", help = "does pretrain task",
        type=utils.str2bool, nargs='?', const=True, default=False
    )
    parser.add_argument(
        "--is_size", help="",
        type=utils.str2bool, nargs='?', const=True, default=False
    )
    parser.add_argument(
        "--is_time", help="",
        type=utils.str2bool, nargs='?', const=True, default=False
    )
    parser.add_argument(
        "--uncertainty", help="Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics",
        type=utils.str2bool, nargs='?', const=True, default=False
    )
    parser.add_argument(
        "--is_tuning", help="does use pretrain-model to tuning",
        type=utils.str2bool, nargs="?", const=True, default=False
    )
    parser.add_argument(
        "--cnn_modules", help="", type=str, default=None,
    )
    parser.add_argument(
        "--attention_modules", help="", type=str, default=None
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
        "--label_mapping_path", help="mapping label to int",
        default=None, type=str,
    )
    parser.add_argument(
        "--params_path", help="parameters' path",
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
    if args.label_mapping_path:
        with open(args.label_mapping_path, "r", encoding="utf-8") as fr:
            label_mapping = json.load(fr)
        print(label_mapping)
        num_labels = len(label_mapping)
        hyperParams["num_labels"] = num_labels
    else:
        label_mapping = None

    hyperParams = utils.params_update(hyperParams, args)
    model = utils.build_model(hyperParams, args.model_name)

    if args.mode == "train":
        log_path = os.path.join(config.log_dir, f"{args.model_name}.{datetime.datetime.now().strftime('%m-%d-%H-%M')}.logs")
        utils.init_logging(log_path)
        logging.info(hyperParams)

        if args.cnn_modules != None:
            train.train(model, label_mapping,
                        args.batch_size, args.num_epochs,
                        args.train_file_path, args.dev_file_path,
                        args.model_name + "_" + args.cnn_modules, args.is_tuning,
                        restore_ckpt = args.restore_ckpt, start_epoch = args.start_epoch,
                        func = utils.str2func(args.cnn_modules)
                        )
        elif args.attention_modules != None:
            train.train(model, label_mapping, args.batch_size, args.num_epochs,
                        args.train_file_path, args.dev_file_path,
                        args.model_name, args.is_tuning,
                        estore_ckpt=args.restore_ckpt, start_epoch = args.start_epoch,
                        func=utils.str2func(args.attention_modules),
                        )
        else:
            train.train(model, label_mapping, args.batch_size, args.num_epochs,
                        args.train_file_path, args.dev_file_path,
                        args.model_name, args.is_tuning,
                        restore_ckpt=args.restore_ckpt, start_epoch = args.start_epoch)

    elif args.mode == "test":
        if args.cnn_modules != None:
            test.test(model, label_mapping, args.batch_size,
                      args.test_file_path, args.model_name + "_" + args.cnn_modules,
                      args.restore_ckpt, args.file_output, func = utils.str2func(args.cnn_modules))
        elif args.attention_modules != None:
            test.test(model, label_mapping, args.batch_size,
                      args.test_file_path, args.model_name,
                      args.restore_ckpt, args.file_output, func=utils.str2func(args.attention_modules))
        else:
            test.test(model, label_mapping, args.batch_size,
                      args.test_file_path, args.model_name,
                      args.restore_ckpt, args.file_output)


