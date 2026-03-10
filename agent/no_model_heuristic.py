from __future__ import annotations

from typing import Iterable
import random

import numpy as np

from gomoku_logic import BOARD_SIZE


def center_prefer_move(actions: Iterable[int]) -> int:
    candidates = [int(a) for a in actions]
    if not candidates:
        raise ValueError("No candidate action")
    center = (BOARD_SIZE - 1) / 2.0
    best_score = None
    best_actions: list[int] = []
    for a in candidates:
        r, c = divmod(a, BOARD_SIZE)
        score = -abs(r - center) - abs(c - center)
        if best_score is None or score > best_score:
            best_score = score
            best_actions = [a]
        elif score == best_score:
            best_actions.append(a)
    return random.choice(best_actions) if best_actions else random.choice(candidates)


def pick_heuristic_move(legal_moves: Iterable[int]) -> tuple[int, np.ndarray]:
    action = center_prefer_move(legal_moves)
    policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    policy[action] = 1.0
    return action, policy
