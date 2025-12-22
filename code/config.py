# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser(description="Pytorch implementation of tilde_L1")

# data
parser.add_argument('--embed_dim', default=128, type=int, help='embedding dimension')
parser.add_argument('--long_lat_embed_dim', default=2, type=int, help='longtitude and latitude embedding dimension')

# Model
parser.add_argument('--type', default=2, type=int, help='type')
parser.add_argument('--batch_size', default=1024, type=int, help='Batch size')
parser.add_argument('--num_epoch', default=5, type=int, help='Number of epochs')
parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
parser.add_argument('--selected_ratio', default=1, type=float, help='selected ratio')
parser.add_argument('--r', default=2, type=int, help='r')
parser.add_argument('--n_output', default=64, type=int, help='output')

# City
parser.add_argument('--city', default='no', type=str, help='city name')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
