import copy
import math
import time

import numpy as np
import torch

import shogi

from pydlshogi.common import *
from pydlshogi.features import *
from pydlshogi.network.policy_value_resnet import *
from pydlshogi.player.base_player import *
from pydlshogi.uct.uct_node import *

C_PUCT = 1.0
CONST_PLAYOUT = 300
RESIGN_THRESHOLD = 0.01
TEMPERATURE = 1.0

def softmax_temperature_with_normalize(logits, temperature):
    """Boltzmann 分布に従う分布にする

    Args:
        logits (list of float in [0, 1]): 指し手の確率分布
        temperature (float): 温度パラメータ

    Returns:
        numpy.ndarray: logits の Boltzmann 分布

    """
    logits /= temperature
    max_logit = max(logits)
    probabilities = np.exp(logits - max_logit)
    probabilities /= sum(probabilities)

    return probabilities

class PlayoutInfo:
    """playout 設定

    Attributes:
        halt (int): 探索打ち切り回数
        count (int): 現在の探索回数

    """
    def __init__(self):
        self.halt = 0
        self.count = 0

class MCTSPlayer(BasePlayer):
    """モンテカルロ木探索プレイヤー

    Attributes:
        modelfile (string): モデルファイルのパス
        model (PolicyValueResnet): 学習モデル
        node_hash (NodeHash): ノード衝突情報
        uct_node (list of UctNode): ノード情報
        po_info (PlayoutInfo): プレイアウト回数管理
        playout (int): 探索回数閾値
        temperature (float): Boltzmann 分布における温度

    """
    def __init__(self):
        super().__init__()
        self.modelfile = r'model/model_policy_value_resnet'
        self.model = None
        self.node_hash = NodeHash()
        self.uct_node = [UctNode() for _ in range(UCT_HASH_SIZE)]
        self.po_info = PlayoutInfo()
        self.playout = CONST_PLAYOUT
        self.temperature = TEMPERATURE

    def select_max_ucb_child(self, board, current_node):
        """UCB の値が最大の手を求める

        UCB = Q + U
        Q = W / N
        U = C_PUCT * P * sqrt(sum(N)) / (1 + N)

        Args:
            board (shogi.Board): 盤面
            current_node (UctNode): ノード

        Returns:
            int: current_node の子ノードのうちに UCB の値が最大の array_index

        """
        child_num = current_node.child_num
        child_win = current_node.child_win
        child_move_count = current_node.child_move_count

        q = np.divide(child_win, child_move_count, out=np.repeat(np.float32(0.5), child_num), where=child_move_count != 0)
        u = np.sqrt(np.float32(current_node.move_count)) / (1 + child_move_count)
        ucb = q + C_PUCT * current_node.nnrate.numpy() * u

        return np.argmax(ucb)

    def expand_node(self, board):
        """board に対してノードを展開する

        衝突情報に対応するノードがあれば, 該当ノードの index を返す
        なければ新規衝突情報ノードを作成し, ノード情報を更新する

        Args:
            board (shogi.Board): 盤面

        Returns:
            int: index

        """
        index = self.node_hash.find_same_hash_index(board.zobrist_hash(), board.turn, board.move_number)
        if not index == UCT_HASH_SIZE:
            return index

        index = self.node_hash.search_empty_index(board.zobrist_hash(), board.turn, board.move_number)

        current_node = self.uct_node[index]
        current_node.move_count = 0
        current_node.win = 0.0
        current_node.value_win = 0.0
        current_node.evaled = False

        current_node.child_move = [move for move in board.legal_moves]
        child_num = len(current_node.child_move)
        current_node.child_num = child_num
        current_node.child_index = [NOT_EXPANDED for _ in range(child_num)]
        current_node.child_move_count = np.zeros(child_num, dtype=np.int32)
        current_node.child_win = np.zeros(child_num, dtype=np.float32)

        if child_num > 0:
            self.eval_node(board, index)
        else: # checkmate
            current_node.value_win = 0.0
            current_node.valued = True

        return index

    def interruption_check(self):
        """探索打ち切り判定

        訪問回数1番目と2番目の差が残り playout 回数より大きい場合は True, そうでなけらば False を返す

        Args:
            board (shogi.Board): 盤面
            current (int): index

        Returns:
            float: 相手の手番の勝率 (= 1 - 現在の勝率)

        """
        child_num = self.uct_node[self.current_root].child_num
        child_move_count = self.uct_node[self.current_root].child_move_count

        rest = self.po_info.halt - self.po_info.count
        second, first = child_move_count[np.argpartition(child_move_count, -2)[:2]]
        return first - second > rest

    def uct_search(self, board, current):
        """UCT (モンテカルロ木)探索

        ゲーム木をたどって、詰み or 未展開のノードで相手の手番の勝率を返す

        Args:
            board (shogi.Board): 盤面
            current (int): index

        Returns:
            float: 相手の手番の勝率 (= 1 - 現在の勝率)

        """
        current_node = self.uct_node[current]
        if current_node.child_num == 0: # checkmate
            return 1.0

        child_move = current_node.child_move
        child_move_count = current_node.child_move_count
        child_index = current_node.child_index
        
        next_index = self.select_max_ucb_child(board, current_node)
        board.push(child_move[next_index])

        if child_index[next_index] == NOT_EXPANDED:
            index = self.expand_node(board)
            child_index[next_index] = index
            child_node = self.uct_node[index]
            result = 1 - child_node.value_win
        else:
            result = self.uct_search(board, child_index[next_index])

        current_node.win += result
        current_node.move_count += 1
        current_node.child_win[next_index] += result
        current_node.child_move_count[next_index] += 1

        board.pop()

        return 1 - result

    def eval_node(self, board, index):
        """board に対してノードを評価する

        self.uct_node[index] の nnrate, value_win を評価し, evaled を済みにする

        Args:
            board (shogi.Board): 盤面
            index (int): index

        """
        eval_features = [make_input_features_from_board(board)]
        x = torch.from_numpy(np.array(eval_features, dtype=np.float32)).to(device)
        with torch.no_grad():
            y1, y2 = self.model(x)
            logits = y1[0].to('cpu')
            value = torch.sigmoid(y2[0]).to('cpu')

        current_node = self.uct_node[index]
        child_num = current_node.child_num
        child_move = current_node.child_move
        color = self.node_hash[index].color

        legal_move_labels = []
        for i in range(child_num):
            legal_move_labels.append(make_output_label(child_move[i], color))
        
        probabilities = softmax_temperature_with_normalize(logits[legal_move_labels], self.temperature)
        current_node.nnrate = probabilities
        current_node.value_win = float(value)
        current_node.evaled = True

    def usi(self):
        print('id name mcts_player')
        print('option name modelfile type string default ' + self.modelfile)
        print('option name playout type spin default ' + str(self.playout) + ' min 100 max 10000')
        print('option name temperature type spin default ' + str(int(self.temperature * 100)) + ' min 10 max 1000')
        print('usiok')

    def setoption(self, option):
        if option[1] == 'modelfile':
            self.modelfile = option[3]
        elif option[1] == 'playout':
            self.playout = int(option[3])
        elif option[1] == 'temperature':
            self.temperature = int(option[3]) / 100

    def isready(self):
        if self.model is None:
            self.model = PolicyValueResnet().to(device)
        else:
            self.model = torch.load(self.modelfile).to(device)
        self.node_hash.initialize()
        print('readyok')

    def go(self):
        if self.board.is_game_over():
            print('bestmove resign')
            return

        self.node_hash.delete_old_hash(self.board, self.uct_node)

        begin_time = time.time()

        self.po_info.count = 0
        self.po_info.halt = self.playout

        self.current_root = self.expand_node(self.board)
        current_node = self.uct_node[self.current_root]
        child_num = current_node.child_num
        child_move = current_node.child_move
        if child_num == 1:
            print('bestmove', child_move[0].usi())
            return

        while self.po_info.count < self.po_info.halt:
            self.po_info.count += 1
            self.uct_search(self.board, self.current_root)
            if self.interruption_check() or not self.node_hash.enough_size:
                break

        finish_time = time.time() - begin_time

        child_move_count = current_node.child_move_count
        if self.board.move_number < 10:
            selected_index = np.random.choice(np.arange(child_num), p=child_move_count / sum(child_move_count))
        else:
            selected_index = np.argmax(child_move_count)

        child_win = current_node.child_win
        for i in range(child_num):
            print('{:3}:{:5} move_count:{:4} nn_rate:{:.5f} win_rate:{:.5f}'.format(
                i, child_move[i].usi(), child_move_count[i], current_node.nnrate[i],
                child_win[i] / child_move_count[i] if child_move_count[i] > 0 else 0))

        best_wp = child_win[selected_index] / child_move_count[selected_index]
        if best_wp < RESIGN_THRESHOLD:
            print('bestmove resign')
            return

        bestmove = child_move[selected_index]
        if best_wp == 1.0:
            cp = 30000
        else:
            cp = int(-math.log(1.0 / best_wp - 1.0) * 600)

        print('info nps {} time {} nodes {} hashfull {} score cp {} pv {}'.format(
            int(current_node.move_count / finish_time),
            int(finish_time * 1000),
            current_node.move_count,
            int(self.node_hash.get_usage_rate() * 1000),
            cp, bestmove.usi()))

        print('bestmove', bestmove.usi())
