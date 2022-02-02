"""棋譜ファイル一覧を訓練用とテスト用に分割する

*_train.txt' と *_test.txt に棋譜ファイル一覧をを分割する

"""

import argparse
import os
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='kifu directory')
    parser.add_argument('filename', type=str, help='train/test file basename')
    parser.add_argument('--ratio', type=float, default=0.9, help='train kifu ratio of the total')
    args = parser.parse_args()

    kifu_list = []
    for root, dirs, files in os.walk(args.dir):
        for file in files:
            kifu_list.append(os.path.join(root, file))
    random.shuffle(kifu_list)
    train_len = int(len(kifu_list) * args.ratio)

    with open(args.filename + '_train.txt', 'w') as f:
        for i in range(train_len):
            f.write(kifu_list[i])
            f.write('\n')

    with open(args.filename + '_test.txt', 'w') as f:
        for i in range(train_len, len(kifu_list)):
            f.write(kifu_list[i])
            f.write('\n')

    print('total kifu num = {}'.format(len(kifu_list)))
    print('train kifu num = {}'.format(train_len))
    print('test kifu num = {}'.format(len(kifu_list) - train_len))
