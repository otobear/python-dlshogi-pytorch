import copy
import math
from threading import Thread, Lock
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
VIRTUAL_LOSS = 1
THREAD_NUM = 4

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

class ParallelMCTSPlayer(BasePlayer):
    """モンテカルロ木探索プレイヤー(並列化)

    Attributes:
        modelfile (string): モデルファイルのパス
        model (PolicyValueResnet): 学習モデル
        node_hash (NodeHash): ノード衝突情報
        uct_node (list of UctNode): ノード情報
        po_info (PlayoutInfo): プレイアウト回数管理
        playout (int): 探索回数閾値
        temperature (float): Boltzmann 分布における温度
        lock_node (list of Lock): ノードロック
        lock_expand (Lock): ノード展開ロック
        lock_po_info (Lock): プレイアウトロック
        current_queue_index (int): 現在参照中のキューインデックス
        features (list of UctNode): 特徴量
        hash_index_queues (list of UctNode): hash index
        current_features (list of UctNode): 現在参照中の特徴量
        current_hash_index_queues (list of UctNode): 現在参照中の hash index

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
        self.lock_node = [Lock() for _ in range(UCT_HASH_SIZE)]
        self.lock_expand = Lock()
        self.lock_po_info = Lock()
        self.current_queue_index = 0
        self.features = [[], []]
        self.hash_index_queues = [[], []]
        self.current_features = self.features[self.current_queue_index]
        self.current_hash_index_queues = self.hash_index_queues[self.current_queue_index]
        self.thread_num = THREAD_NUM

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
            self.current_features.append(make_input_features_from_board(board))
            self.current_hash_index_queue.append(index)
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

    def parallel_uct_search(self):
        while True:
            with self.lock_po_info:
                self.po_info.count += 1
            board = copy.deepcopy(self.board)
            self.uct_search(board, self.current_root)
            with self.lock_po_info:
                with self.lock_node[self.current_root]:
                    if self.po_info.count >= self.po_info.halt or self.interruption_check() or not self.node_hash.enough_size:
                        return

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

        while not current_node.evaled:
            time.sleep(0.000001)

        self.lock_node[current].acquire()
        
        next_index = self.select_max_ucb_child(board, current_node)
        board.push(child_move[next_index])

        current_node.move_count += VIRTUAL_LOSS
        child_move_count[next_index] += VIRTUAL_LOSS

        if child_index[next_index] == NOT_EXPANDED:
            with self.lock_expand:
                index = self.expand_node(board)
            child_index[next_index] = index

            self.lock_node[current].release()

            child_node = self.uct_node[index]
            while not child_node.evaled:
                time.sleep(0.000001)
            result = 1 - child_node.value_win
        else:
            self.lock_node[current].release()

            result = self.uct_search(board, child_index[next_index])

        with self.lock_node[current]:
            current_node.win += result
            current_node.move_count += 1 - VIRTUAL_LOSS
            current_node.child_win[next_index] += result
            current_node.child_move_count[next_index] += 1 - VIRTUAL_LOSS

        return 1 - result

    def clear_eval_queue(self):
        """que をクリアする

        self.features, self.hash_index_queues をクリアし, current_features と current_hash_index_queue を初期化する

        """
        self.current_queue_index = 0
        for i in range(2):
            self.features[i].clear()
            self.hash_index_queues[i].clear()
        self.current_features = self.features[self.current_queue_index]
        self.current_hash_index_queue = self.hash_index_queues[self.current_queue_index]

    def eval_node(self):
        """que に溜まっているノードを評価する

        """
        enough_batch_size = False

        while True:
            if not self.running:
                break

            self.lock_expand.acquire()
            if len(self.current_hash_index_queue) == 0:
                self.lock_expand.release()
                time.sleep(0.000001)
                continue

            if self.running and not enough_batch_size and len(self.current_hash_index_queue) < self.thread_num * 0.5:
                self.lock_expand.release()
                time.sleep(0.000001)
                enough_batch_size = True
                continue

            enough_batch_size = False

            eval_features = self.current_features
            eval_hash_index_queue = self.current_hash_index_queue
            self.current_queue_index = self.current_queue_index ^ 1
            self.current_features = self.features[self.current_queue_index]
            self.current_hash_index_queue = self.hash_index_queues[self.current_queue_index]
            self.current_features.clear()
            self.current_hash_index_queue.clear()
            self.lock_expand.release()

            x = torch.from_numpy(np.array(eval_features, dtype=np.float32)).to(device)
            with torch.no_grad():
                y1, y2 = self.model(x)
                logits_batch = y1.to('cpu')
                values_batch = torch.sigmoid(y2).to('cpu')

            for index, logits, value in zip(eval_hash_index_queue, logits_batch, values_batch):
                self.lock_node[index].acquire()

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
                self.lock_node[index].release()
    
    def usi(self):
        print('id name parallel_mcts_player')
        print('option name modelfile type string default ' + self.modelfile)
        print('option name playout type spin default ' + str(self.playout) + ' min 100 max 10000')
        print('option name temperature type spin default ' + str(int(self.temperature * 100)) + ' min 10 max 1000')
        print('option name thread type spin default ' + str(int(self.temperature * 100)) + ' min 10 max 1000')
        print('usiok')

    def setoption(self, option):
        if option[1] == 'modelfile':
            self.modelfile = option[3]
        elif option[1] == 'playout':
            self.playout = int(option[3])
        elif option[1] == 'temperature':
            self.temperature = int(option[3]) / 100
        elif option[1] == 'thread':
            self.thread = int(option[3])

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

        self.clear_eval_queue()

        self.current_root = self.expand_node(self.board)
        current_node = self.uct_node[self.current_root]
        child_num = current_node.child_num
        child_move = current_node.child_move
        if child_num == 1:
            print('bestmove', child_move[0].usi())
            return

        self.running = True

        th_nn = Thread(target=self.eval_node)
        th_nn.start()

        threads = []
        for i in range(self.thread_num):
            th = Thread(target=self.parallel_uct_search)
            th.start()
            threads.append(th)

        for th in threads:
            th.join()

        self.running = False

        th_nn.join()

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
