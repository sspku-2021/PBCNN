# -*- coding: utf-8 -*-

from scapy.all import hex_bytes, bytes_hex
from scapy.layers.l2 import Dot3
from scapy.all import bytes_encode


def bytes2str(info):
    return bytes.decode(bytes_hex(info))


def str2bytes(info):
    return Dot3(hex_bytes(str.encode(info)))


def standardzation(input, mu, sigma):
    return (input - mu) / sigma


def standardzation_reverse(input, mu, sigma):
    return input * sigma + mu


def pkt_to_pixel(info):
    return [int(b) for b in bytes_encode(info)]


