"""
Core Gomoku board logic shared by GUI and AlphaZero training.
Board is 20x20, win condition is five in a row.
Stones: 1 for black, -1 for white, 0 for empty. Black goes first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

BOARD_SIZE = 20
WIN_LENGTH = 5
BLACK = 1
WHITE = -1
EMPTY = 0


@dataclass
class MoveResult:
    board: np.ndarray
    player: int
    winner: Optional[int]
    finished: bool


class Gomoku:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.player = BLACK  # Black starts
        self.winner: Optional[int] = None
        self.finished = False
        self.last_move: Optional[Tuple[int, int]] = None

    def clone(self) -> "Gomoku":
        clone = Gomoku()
        clone.board = self.board.copy()
        clone.player = self.player
        clone.winner = self.winner
        clone.finished = self.finished
        clone.last_move = self.last_move
        return clone

    def reset(self):
        self.board.fill(EMPTY)
        self.player = BLACK
        self.winner = None
        self.finished = False
        self.last_move = None

    def legal_moves(self) -> List[int]:
        if self.finished:
            return []
        empties = np.argwhere(self.board == EMPTY)
        return [r * BOARD_SIZE + c for r, c in empties]

    def step(self, action: int) -> MoveResult:
        if self.finished:
            raise ValueError("Game already finished")
        r, c = divmod(action, BOARD_SIZE)
        if self.board[r, c] != EMPTY:
            raise ValueError("Illegal move")
        self.board[r, c] = self.player
        self.last_move = (r, c)
        if self._check_winner(r, c):
            self.winner = self.player
            self.finished = True
        elif not (self.board == EMPTY).any():
            self.winner = 0  # draw
            self.finished = True
        self.player = -self.player
        return MoveResult(self.board.copy(), self.player, self.winner, self.finished)

    def _check_winner(self, r: int, c: int) -> bool:
        # Directions: (dr, dc)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        player = self.board[r, c]
        for dr, dc in directions:
            count = 1
            count += self._count_dir(r, c, dr, dc, player)
            count += self._count_dir(r, c, -dr, -dc, player)
            if count >= WIN_LENGTH:
                return True
        return False

    def _count_dir(self, r: int, c: int, dr: int, dc: int, player: int) -> int:
        cnt = 0
        nr, nc = r + dr, c + dc
        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and self.board[nr, nc] == player:
            cnt += 1
            nr += dr
            nc += dc
        return cnt

    def canonical_board(self) -> np.ndarray:
        """
        Returns board from current player's perspective as 2 planes.
        plane 0: current player's stones (1/0), plane 1: opponent stones.
        """
        cur = (self.board == self.player).astype(np.float32)
        opp = (self.board == -self.player).astype(np.float32)
        return np.stack([cur, opp], axis=0)

    def result_for_player(self, player: int) -> float:
        if self.winner is None:
            raise ValueError("Game not finished")
        if self.winner == 0:
            return 0.0
        return 1.0 if self.winner == player else -1.0
