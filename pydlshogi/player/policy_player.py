import numpy as np
import torch
import torch.nn.functional as F

import shogi

from pydlshogi.common import *
from pydlshogi.features import *
from pydlshogi.network.policy import *
from pydlshogi.player.base_player import *

def greedy(logits):
    return logits.index(max(logits))

def boltzmann(logits, temperature):
    logits /= temperature
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    
    return np.random.choice(len(logits), p=probabilities)

class PolicyPlayer(BasePlayer):
    def __init__(self):
        super().__init__()
        self.modelfile = r'model/model_policy'
        self.model = None

    def usi(self):
        print('id name policy_player')
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

        features = make_input_features_from_board(self.board)
        x = torch.from_numpy(np.array([features], dtype=np.float32)).to(device)

        with torch.no_grad():
            y = self.model(x).to('cpu')
            logits = y[0].tolist()
            probabilities = F.softmax(y, dim=-1)[0].tolist()

        legal_moves = []
        legal_logits = []
        for move in self.board.legal_moves:
            label = make_output_label(move, self.board.turn)
            legal_moves.append(move)
            legal_logits.append(logits[label])
            print('info string {:5} : {:.5f}'. format(move.usi(), probabilities[label]))

        selected_index = greedy(legal_logits)
        # selected_index = boltzmann(np.array(legal_logits, dtype=np.float32), 0.5)
        best_move = legal_moves[selected_index]
        print('bestmove', best_move.usi())
