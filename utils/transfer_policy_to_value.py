"""policy network から value network へ転移学習を行う

policy model から value model へ同じ名前の parameter をコピーする

"""

import torch

from pydlshogi.network.policy import *
from pydlshogi.network.value import *

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('policy_model', type=str, help='trained policy model file path')
    parser.add_argument('value_model', type=str, help='value model file path to save')
    args = parser.parse_args()

    policy_model = PolicyNetwork()
    value_model = ValueNetwork()

    print('Load policy model from', args.policy_model)
    policy_model = torch.load(args.policy_model)

    print('Value model params')
    value_dict = {}
    for path, param in value_model.named_parameters():
        print(path, param.shape)
        value_dict[path] = param

    print('Policy model params')
    for path, param in policy_model.named_parameters():
        print(path, param.shape)
        if path in value_dict:
            value_dict[path] = param.data

    print('Save the model')
    torch.save(value_model, args.value_model)
