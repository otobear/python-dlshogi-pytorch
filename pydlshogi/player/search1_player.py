import numpy as np
import torch
import torch.nn.functional as F

import shogi

from pydlshogi.common import *
from pydlshogi.features import *
from pydlshogi.network.policy import *
from pydlshogi.player.base_player import *

def greedy(logits):
    return np.argmax(logits)

def boltzmann(logits, temperature):
    logits /= temperature
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    
    return np.random.choice(len(logits), p=probabilities)

class Search1Player(BasePlayer):
    def __init__(self):
        super().__init__()
        self.modelfile = r'model/model_value'
        self.model = None

    def usi(self):
        print('id name search1_player')
        print('option name modelfile type string default ' + self.modelfile)
        print('usiok')

    def setoption(self, option):
        if option[1] == 'modelfile':
            self.modelfile = option[3]

    def isready(self):
        self.model = torch.load(self.modelfile).to(device)
        print('readyok')

    def go(self):
        if self.board.is_game_over():
            print('bestmove resign')
            return

        legal_moves = []
        features = []
        for move in self.board.legal_moves:
            legal_moves.append(move)
            self.board.push(move)
            features.append(make_input_features_from_board(self.board))
            self.board.pop()

        x = torch.from_numpy(np.array(features, dtype=np.float32)).to(device)

        with torch.no_grad():
            y = -self.model(x).to('cpu')
            logits = y.reshape(-1)
            probabilities = torch.sigmoid(y).reshape(-1)

        for i, move in enumerate(legal_moves):
            print('info string {:5} : {:.5f}'.format(move.usi(), probabilities[i]))

        selected_index = greedy(logits)
        # selected_index = boltzmann(np.array(logits, dtype=np.float32), 0.5)
        best_move = legal_moves[selected_index]
        print('bestmove', best_move.usi())
