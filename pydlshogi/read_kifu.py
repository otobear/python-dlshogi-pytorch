"""棋譜ファイル一覧から学習データを生成する

"""

import shogi
import shogi.CSA
import copy

from pydlshogi.features import *

def read_kifu(kifu_list_file):
    """棋譜ファイル一覧から学習データを生成する

    Args:
        kifu_list_file(list of string): 棋譜ファイル一覧

    Returns:
        list of tuple(5): 学習データ. list[(駒配置, 占有座標, 持ち駒, 指し手, 勝者)]

    """
    positions = []
    with open(kifu_list_file, 'r') as f:
        for line in f.readlines():
            filepath = line.rstrip('\r\n')
            kifu = shogi.CSA.Parser.parse_file(filepath)[0]
            win_color = shogi.BLACK if kifu['win'] == 'b' else shogi.WHITE
            board = shogi.Board()
            for move in kifu['moves']:
                if board.turn == shogi.BLACK:
                    piece_bb = copy.deepcopy(board.piece_bb)
                    occupied = copy.deepcopy((board.occupied[shogi.BLACK], board.occupied[shogi.WHITE]))
                    pieces_in_hand = copy.deepcopy((board.pieces_in_hand[shogi.BLACK], board.pieces_in_hand[shogi.WHITE]))
                else:
                    piece_bb = [bb_rotate_180(bb) for bb in board.piece_bb]
                    occupied = (bb_rotate_180(board.occupied[shogi.WHITE]), bb_rotate_180(board.occupied[shogi.BLACK]))
                    pieces_in_hand = copy.deepcopy((board.pieces_in_hand[shogi.WHITE], board.pieces_in_hand[shogi.BLACK]))
                move_label = make_output_label(shogi.Move.from_usi(move), board.turn)
                win = 1 if win_color == board.turn else 0
                positions.append((piece_bb, occupied, pieces_in_hand, move_label, win))
                board.push_usi(move)

    return positions
