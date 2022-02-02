"""マルチタスク(方策, 価値)ネットワークの学習

"""

import argparse
import random
import pickle
import os
import re
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from pydlshogi.common import *
from pydlshogi.network.policy_value import PolicyValueNetwork
from pydlshogi.features import *
from pydlshogi.read_kifu import *

parser = argparse.ArgumentParser()
parser.add_argument('kifulist_train', type=str, help='train kifu list')
parser.add_argument('kifulist_test', type=str, help='test kifu list')
parser.add_argument('--batchsize', '-b', type=int, default=32, help='mini-batch size')
parser.add_argument('--test_batchsize', type=int, default=512, help='test mini-batch size')
parser.add_argument('--epoch', '-e', type=int, default=1, help='epoch times')
parser.add_argument('--model', type=str, default='model/model_policy_value', help='model file save path')
parser.add_argument('--state', type=str, default='model/state_policy_value', help='state file save path')
parser.add_argument('--initmodel', type=str, default='', help='resumed model file path')
parser.add_argument('--resume', '-r', type=str, default='', help='resumed state file path')
parser.add_argument('--log', type=str, default=None, help='log file path')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--eval_interval', '-i', type=int, default=1000, help='eval interval')
parser.add_argument('--tensorboard_writer', '-w', type=str, default='tensorboard', help='tensorboard save path')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                    filename=args.log, level=logging.DEBUG)

model = PolicyValueNetwork()
model.to(device)
if args.initmodel:
    logging.info('Load model from {}'.format(args.initmodel))
    model = torch.load(args.initmodel).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
criterion_policy = nn.CrossEntropyLoss()
criterion_value = nn.BCEWithLogitsLoss()
if args.resume:
    logging.info('Load optimizer state from {}'.format(args.resume))
    optimizer = torch.load(args.resume).to(device)

if not os.path.join(args.tensorboard_writer):
    os.makedirs(args.tensorboard_writer)
writer_iter = SummaryWriter(args.tensorboard_writer)

logging.info('Read kifu start')
train_pickle_filename = re.sub(r'\..*?$', '', args.kifulist_train) + '.pickle'
if os.path.exists(train_pickle_filename):
    with open(train_pickle_filename, 'rb') as f:
        positions_train = pickle.load(f)
    logging.info('Load train pickle')
else:
    positions_train = read_kifu(args.kifulist_train)
    with open(train_pickle_filename, 'wb') as f:
        pickle.dump(positions_train, f, pickle.HIGHEST_PROTOCOL)
    logging.info('Save train pickle')

test_pickle_filename = re.sub(r'\..*?$', '', args.kifulist_test) + '.pickle'
if os.path.exists(test_pickle_filename):
    with open(test_pickle_filename, 'rb') as f:
        positions_test = pickle.load(f)
    logging.info('Load test pickle')
else:
    positions_test = read_kifu(args.kifulist_test)
    with open(test_pickle_filename, 'wb') as f:
        pickle.dump(positions_test, f, pickle.HIGHEST_PROTOCOL)
    logging.info('Save test pickle')
logging.info('Read kifu end')
logging.info('Train position num = {}'.format(len(positions_train)))
logging.info('Test position num = {}'.format(len(positions_test)))

def mini_batch(positions, i, batchsize):
    """mini batchを生成する

    学習データからmini batchを生成する

    Args:
        positions (list of tuple(5)): 学習データ
        i (int): mini batchスタートインデックス
        batchsize (int): batch size

    Returns:
        tuple(3): (入力特徴量配列, 出力特徴量(指し手)配列)

    """
    mini_batch_data = []
    mini_batch_move = []
    mini_batch_win = []
    for b in range(batchsize):
        features, move, win = make_features(positions[i + b])
        mini_batch_data.append(features)
        mini_batch_move.append(move)
        mini_batch_win.append(win)

    return (torch.from_numpy(np.array(mini_batch_data, dtype=np.float32)),
            torch.from_numpy(np.array(mini_batch_move, dtype=np.int64)),
            torch.from_numpy(np.array(mini_batch_win, dtype=np.float32).reshape(-1, 1)))

def mini_batch_for_test(positions, batchsize):
    """テスト用mini batchを生成する

    テストデータからmini batchを生成する

    Args:
        positions (list of tuple(5)): 学習データ
        batchsize(int): batch size

    Returns:
        tuple(3): (テスト用入力特徴量配列, テスト用出力特徴量(指し手)配列)

    """
    mini_batch_data = []
    mini_batch_move = []
    mini_batch_win = []
    for b in range(batchsize):
        features, move, win = make_features(random.choice(positions))
        mini_batch_data.append(features)
        mini_batch_move.append(move)
        mini_batch_win.append(win)

    return (torch.from_numpy(np.array(mini_batch_data, dtype=np.float32)),
            torch.from_numpy(np.array(mini_batch_move, dtype=np.int64)),
            torch.from_numpy(np.array(mini_batch_win, dtype=np.float32).reshape(-1, 1)))

def accuracy(labels, outputs):
    predicts = torch.argmax(outputs, dim=1)
    return torch.mean((labels == predicts).float())

def binary_accuracy(labels, outputs):
    predicts = outputs >= 0
    return torch.mean((labels == predicts).float())

logging.info('Start training')
itr = 0
sum_loss = 0
current_step = 0
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
for e in range(args.epoch):
    positions_train_shuffled = random.sample(positions_train, len(positions_train))
    itr_epoch = 0
    sum_loss_epoch = 0
    for i in range(0, len(positions_train) - args.batchsize, args.batchsize):
        mb = mini_batch(positions_train_shuffled, i, args.batchsize)
        x, t1, t2 = mb[0].to(device), mb[1].to(device), mb[2].to(device)
        output = model(x)
        y1, y2 = output[0].to(device), output[1].to(device)

        optimizer.zero_grad()
        loss = criterion_policy(y1, t1) + criterion_value(y2, t2)
        loss.backward()
        optimizer.step()
        current_step += 1

        itr += 1
        sum_loss += loss.item()
        itr_epoch += 1
        sum_loss_epoch += loss.item()

        if current_step % args.eval_interval == 0:
            mb = mini_batch_for_test(positions_test, args.test_batchsize)
            x, t1, t2 = mb[0].to(device), mb[1].to(device), mb[2].to(device)
            output = model(x)
            y1, y2 = output[0].to(device), output[1].to(device)

            logging.info('epoch = {}, iteration = {}, loss = {}, accuracy = {}, {}'.format(
                e + 1, current_step, sum_loss / itr, accuracy(t1, y1), binary_accuracy(t2, y2)))

            writer_iter.add_scalar('loss/train', loss.item(), itr_epoch)
            for name, param in model.named_parameters():
                writer_iter.add_histogram('batch_' + name + '_param', param, i)
                writer_iter.add_histogram('batch_' + name + '_grad', param.grad, i)

            itr = 0
            sum_loss = 0

    logging.info('Validate test data')
    itr_test = 0
    sum_test_accuracy_policy = 0
    sum_test_accuracy_value = 0
    with torch.no_grad():
        for i in range(0, len(positions_test) - args.batchsize, args.batchsize):
            mb = mini_batch(positions_train_shuffled, i, args.batchsize)
            x, t1, t2 = mb[0].to(device), mb[1].to(device), mb[2].to(device)
            output = model(x)
            y1, y2 = output[0].to(device), output[1].to(device)

            itr_test += 1
            sum_test_accuracy_policy += accuracy(t1, y2)
            sum_test_accuracy_value += binary_accuracy(t1, y2)

        logging.info('epoch = {}, iteration = {}, train loss avr = {}, test accuracy = {} {}'.format(
            e + 1, current_step, sum_loss_epoch / itr_epoch, sum_test_accuracy_policy / itr_test, sum_test_accuracy_value / itr_test))

logging.info('Save the model')
torch.save(model, args.model)
torch.save(optimizer, args.state)
