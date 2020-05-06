import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import argparse
from tensorboardX import SummaryWriter

sys.path.append('../')
sys.path.append('../src')
sys.path.append('../model')
sys.path.append('../data')
sys.path.append('../config')

import NCF
import DataUtils as DU
import config

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
    '--dropout',
    type=int,
    default=0.0,
    help='dropout rate for the MLP layers'
)
parser.add_argument(
    '--factor_num',
    type=int,
    default=32,
    help='predictive factors numbers in the model'
)
parser.add_argument(
    '--mlplayer_num',
    type=int,
    default=5,
    help='num of layers for MLP'
)
parser.add_argument(
    '--num_neg',
    type=int,
    default=2,
    help='randomly sample negative data for each tuple in the train dataset.[ONLY IN TRAIN]'
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


train_data, test_data, traing_rating, test_rating, train_label, test_label, test_with_neg, user_num, item_num, train_mat = DU.load_data()

# train_data, _, traing_rating, _, train_label, _, test_with_neg, user_num, item_num, train_mat = DU.load_data()


train_dataset = DU.NCFDataSet(
    train_data, args.num_neg, train_mat, item_num, train_label, test_label, True)
# test_dataset = DU.NCFDataSet(
#     test_data, 0, train_mat, item_num, train_label, test_label, False)
test_with_neg_dataset = DU.NCFDataSet(
    test_with_neg, 0, train_mat, item_num, train_label, test_label, False)

# This need to be done by multiprocessing using cuda
# train_dataloader = data.DataLoader(
#     train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4)

# To avoid multiprocessing, can simply get rid of the num_workers varaible.
train_dataloader = data.DataLoader(
    train_dataset, batch_size=args.batchsize, shuffle=True)

# test_dataloader = data.DataLoader(
#     test_dataset, batch_size=32, shuffle=False, num_workers=0)
# NOTE: each batch is actually for exactly 1 user.
test_with_neg_dataloader = data.DataLoader(
    test_with_neg_dataset, batch_size=100, shuffle=False)

if config.config['model_cur'] == 'NeuMF-pre':
    assert os.path.exists(config.config['GMF_model']), 'lack of pre-trained GMF model'
    assert os.path.exists(config.config['MLP_model']), 'lack of pre-trained MLP model'
    GMF_model = torch.load(config.config['GMF_model'])
    MLP_model = torch.load(config.config['MLP_model'])
else:
    GMF_model = None
    MLP_model = None

# Device CPU / GPU
if args.device is not None:
    if args.device.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(args.device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        # logger.warning(
        #     '{} GPU is not available, running on CPU'.format(__name__))
        print(
            'Warning: {} GPU is not available, running on CPU'
            .format(__name__))
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
# logger.info('{} Using device: {}'.format(__name__, device))
print('{} Using device: {}'.format(__name__, device))

# Seeding
if args.seed is not None:
    # logger.info('{} Setting random seed'.format(__name__))
    print('{} Setting random seed'.format(__name__))
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

model = NCF.NCF(user_num, item_num, args.factor_num, args.num_layers, 
				args.dropout, config.config['model_cur'], GMF_model, MLP_model).to(device)

# loss_function = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

best_HitRate = 0.0
best_HitRate_NDCG = 0.0 # Without normalization
best_HitRate_epoch = 0

# add log writer for tensorboardX
writer = SummaryWriter('log')

for epoch in range(args.epoch_num):
    model.train()
    train_loader.dataset.insert_neg_for_train()
    running_loss = 0.0
    for idx, data in enumerate(train_dataloader):
        user, item, label = data[0].to(device), data[1].to(device), data[2].to(device)
        # zero the param gradient
        optimizer.zero_grad()
        # forward + backward + optimize
        output = model(user, item)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        # print statistics
        # 200 - train
        running_loss += loss.item()
        if (idx + 1) % 200 == 0:
            print(
                '[%d, %5d] loss: %.3f'
                    % (epoch + 1, idx + 1, running_loss / 200)
            )
            # Add writer to record the traning loss
            niter = epoch * len(train_dataloader) + idx + 1
            writer.add_scalar('Train/Loss', running_loss / 200, niter)
            running_loss = 0.0

        # It might be better to add validation

    print('Finished Training at epoch {}\n Start testing...'.format(epoch))

    model.eval()

    # Save model periodly
    if (epoch + 1) % args.save_model_freq == 0:
        model_state = {
            'state_dict': model.state_dict()
        }
        torch.save(
            model_state, '{0}/train_done_{1}.pth'
            .format(args.save_dir, epoch + 1)
        )
    
    hit_at_k = 0.0
    ndcg_at_k = 0.0
    counter = 0
    # Calculate the HIT@10 and NDCG@10 metric
    for idx, data in enumerate(test_dataloader):
        user, item, label = data[0].to(device), data[1].to(device), data[2].to(device)
        output = model(user, item)
        score, indices = torch.topk(output, 10)
        # A list of the topk items predicted by the model
        topk_items = torch.take(item, indices).cpu().numpy().tolist()
        # Get the real value of the ground truth item
        positve_item = item[0].item()
        
        counter += 1

        # Compute the HIT@10 metric
        if positve_item in topk_items:
            hit_at_k += 1.0
        else:
            pass

        # Compute the NDCG@10 metric
        if positve_item in topk_items:
            rank = topk_items.index(positve_item)
            ndcg_at_k += np.reciprocal(np.log2(rank + 2))
        else:
            pass
    
    hit_at_k /= counter
    ndcg_at_k /= counter

    print('Finished Testing at epoch {}\t HIT@10: {}\t NDCG@10: {}\n'.format(epoch, hit_at_k, ndcg_at_k))

    if hit_at_k > best_HitRate:
        model_state = {
            'state_dict': model.state_dict()
        }
        torch.save(
            model_state, '{0}/best_hit_so_far_{1}.pth'
            .format(args.save_dir, epoch + 1)
        )
        best_HitRate = hit_at_k
        best_HitRate_NDCG = ndcg_at_k
        best_HitRate_epoch = epoch

print('Finished.\nBest HIT@10: {}\t Best NDCG@10: {}\t Best epoch \n'.format(best_HitRate, best_HitRate_NDCG, best_HitRate_epoch))
