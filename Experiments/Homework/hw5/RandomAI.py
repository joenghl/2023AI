import copy
from ChessBoard import *
import numpy as np


class RandomAI(object):
    def __init__(self, computer_team):
        self.team = computer_team

    def get_next_step(self, chessboard: ChessBoard):
        solutions = []
        for chess in chessboard.get_chess():
            if chess.team == self.team:
                positions = chessboard.get_put_down_position(chess)
                for pos in positions:
                    x, y = pos
                    solutions.append((chess.row, chess.col, x, y))
        rand_int = np.random.randint(0, len(solutions))
        return solutions[rand_int]
