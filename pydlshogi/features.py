"""特徴量生成メソッドの集め

学習データ/局面から入力特徴量を生成する
指し手から出力特徴量を生成する

"""

import numpy as np
import shogi
import copy

from pydlshogi.common import *

def make_input_features(piece_bb, occupied, pieces_in_hand):
    """学習データから入力特徴量を生成する

    入力特徴量次元数: 盤上8, 盤上成り駒6, 持ち駒34(歩18, 香4, 桂4, 銀4, 金4, 角2, 飛2), 合計52. 先後合計104

    Args:
        piece_bb(shogi.Board.piece_bb): 駒ごとの配置(bit)
        occupied(shgoi.Board.occupied): 手番ごとの占有座標(bit)
        pieces_in_hand(shogi.Board.pieces_in_hand): 手番ごとの持ち駒(dict)

    Returns:
        list[104 * list[9 * list[9]]]: 入力特徴量

    """
    features = []
    for color in shogi.COLORS: # range(0, 2)
        for piece_type in shogi.PIECE_TYPES_WITH_NONE[1:]: # range(1, 15)
            # board pieces
            bb = piece_bb[piece_type] & occupied[color]
            feature = np.zeros(9 * 9)
            for pos in shogi.SQUARES: # range(0, 81)
                if bb & shogi.BB_SQUARES[pos] > 0: # [1, 2, 4, ..., 2^80]
                    feature[pos] = 1
            features.append(feature.reshape((9, 9)))
        for piece_type in range(1, 8):
            # pieces in hand
            # TODO: refactor
            for n in range(shogi.MAX_PIECES_IN_HAND[piece_type]): # [0, 18, 4, 4, 4, 4, 2, 2, 0, 0, 0, 0, 0, 0, 0]
                if piece_type in pieces_in_hand[color] and n < pieces_in_hand[color][piece_type]:
                    feature = np.ones(9 * 9)
                else:
                    feature = np.zeros(9 * 9)
                features.append(feature.reshape((9, 9)))

    return features

def make_input_features_from_board(board):
    """盤面から入力特徴量を生成する

    Args:
        board(shogi.Board): 盤面

    Returns:
        list[104 * list[9 * list[9]]]: 入力特徴量

    """
    if board.turn == shogi.BLACK:
        piece_bb = board.piece_bb
        occupied = (board.occupied[shogi.BLACK], board.occupied[shogi.WHITE])
        pieces_in_hand = (board.pieces_in_hand[shogi.BLACK], board.pieces_in_hand[shogi.WHITE])
    else:
        piece_bb = [bb_rotate_180(bb) for bb in board.piece_bb]
        occupied = (bb_rotate_180(board.occupied[shogi.WHITE]), bb_rotate_180(board.occupied[shogi.BLACK]))
        pieces_in_hand = (board.pieces_in_hand[shogi.WHITE], board.pieces_in_hand[shogi.BLACK])

    return make_input_features(piece_bb, occupied, pieces_in_hand)

def make_output_label(move, color):
    """指し手から出力特徴量を生成する

    方向: 移動10, 打ち7, 成り10, 合27
    位置: 81
    合計: 27 * 81 = 2187

    Args:
        move(shogi.Move.Move): 指し手
        color(shogi.COLORS): 手番

    Returns:
        int: 指し手ラベル

    """
    move_to = move.to_square
    move_from = move.from_square

    if color == shogi.WHITE:
        move_to = SQUARES_R180[move_to]
        if move_from is not None:
            move_from = SQUARES_R180[move_from]

    if move_from is not None:
        # board pieces
        to_y, to_x = divmod(move_to, 9)
        from_y, from_x = divmod(move_from, 9)
        dir_x = to_x - from_x
        dir_y = to_y - from_y
        if dir_y < 0 and dir_x == 0:
            move_direction = UP
        elif dir_y == -2 and dir_x == -1:
            move_direction = UP2_LEFT
        elif dir_y == -2 and dir_x == 1:
            move_direction = UP2_RIGHT
        elif dir_y < 0 and dir_x < 0:
            move_direction = UP_LEFT
        elif dir_y < 0 and dir_x > 0:
            move_direction = UP_RIGHT
        elif dir_y == 0 and dir_x < 0:
            move_direction = LEFT
        elif dir_y == 0 and dir_x > 0:
            move_direction = RIGHT
        elif dir_y > 0 and dir_x == 0:
            move_direction = DOWN
        elif dir_y > 0 and dir_x < 0:
            move_direction = DOWN_LEFT
        elif dir_y > 0 and dir_x > 0:
            move_direction = DOWN_RIGHT

        if move.promotion:
            move_direction = MOVE_DIRECTION_PROMOTED[move_direction]
    else:
        # pieces in hand
        move_direction = len(MOVE_DIRECTION) + move.drop_piece_type - 1

    move_label = 9 * 9 * move_direction + move_to

    return move_label

def make_features(position):
    """学習データから入力特徴量と出力特徴量を生成する

    Args:
        position(tuple(5)): 特徴量(駒配置, 占有座標, 持ち駒, 指し手, 勝者)

    Returns:
        tuple(3): (入力特徴量, 指し手, 勝者)

    """
    piece_bb, occupied, pieces_in_hand, move, win = position
    features = make_input_features(piece_bb, occupied, pieces_in_hand)

    return (features, move, win)
