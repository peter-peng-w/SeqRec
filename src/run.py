import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from tensorboardX import SummaryWriter

sys.path.append('../')
sys.path.append('../src')
sys.path.append('../model')
sys.path.append('../data')
sys.path.append('../config')

from model.NCF import NCF

parser = argparse.ArgumentParser(description="Setting series hyperparamter for training")
parser.add_argument(
    '-d',
    '--device',
    type=str,
    default='cuda:0',
    help='using GPU and cuda to train and also indicate the GPU number'
)
parser.add_argument(
    '-s',
    '--seed',
    type=int,
    default=42,
    help='indicate the seed number while doing random to gain reproductivity'
)
parser.add_argument(
    '--batchsize',
    type=int,
    default=256,
    help='batch size for traning'
)
parser.add_argument(
    '--train_root',
    type=str,
    default='../data/train',
    help='path to the train data (root directory of train data)'
)
parser.add_argument(
    '--test_root',
    type=str,
    default='../data/test',
    help='path to the test data (root directory of test data)'
)
parser.add_argument(
    '--save_dir',
    type=str,
    default='../expr',
    help='the path to save model'
)
parser.add_argument(
    '--save_model_freq',
    type=int,
    default=2,
    help='Frequency of saving model, per epoch'
)
parser.add_argument(
    '--epoch_num',
    type=int,
    default=20,
    help='The number of epoch'
)
args = parser.parse_args()
print(args)

