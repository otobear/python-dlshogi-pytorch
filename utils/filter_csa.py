"""以下の条件を満たさない棋譜を削除する

1. 手数 >= 50
2. 両方 rating >= 2500
3. 投了で終局

"""

import argparse
import os
import re
import statistics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, help='kifu directory')
    args = parser.parse_args()

    def find_all_files(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                yield os.path.join(root, file)

    ptn_rate = re.compile(r"^'(black|white)_rate:.*:(.*)$")
    kifu_count = 0
    rates = []
    for filepath in find_all_files(args.dir):
        rate = {}
        move_len = 0
        toryo = False
        for line in open(filepath, 'r', encoding='utf-8'):
            line = line.strip()
            m = ptn_rate.match(line)
            if m:
                rate[m.group(1)] = float(m.group(2))
            if line[:1] in ('+', '-'):
                move_len += 1
            if line == '%TORYO':
                toryo = True

        if not toryo or move_len <= 50 or len(rate) < 2 or min(rate.values()) < 2500:
            os.remove(filepath)
        else:
            kifu_count += 1
            rates.extend([_ for _ in rate.values()])

    print('kifu count: ', kifu_count)
    print('rate mean: ', statistics.mean(rates))
    print('rate median: ', statistics.median(rates))
    print('rate max: ', max(rates))
    print('rate min: ', min(rates))
