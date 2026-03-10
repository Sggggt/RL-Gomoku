"""
Pattern utilities for tactical motif shaping and supervised labeling.
"""

from __future__ import annotations

from typing import Dict
import numpy as np


PATTERN_CLASSES = [
    "live_three",
    "sleep_three",
    "live_four",
    "rush_four",
    "double_three",
    "four_three",
    "double_four",
    "five",
    "three_plus_one",
    "two_plus_two",
    "two_plus_one",
    "double_three_plus_one",
    "double_two_plus_two",
    "double_two_plus_one",
    "two_plus_one_three_plus_one",
    "live_three_three_plus_one",
    "live_three_two_plus_one",
    "live_three_two_plus_two",
    "two_plus_one_two_plus_two",
]

ATTACK_WEIGHTS: Dict[str, float] = {
    "live_three": 2.0,
    "sleep_three": 1.0,
    "live_four": 6.0,
    "rush_four": 4.0,
    "double_three": 6.0,
    "four_three": 10.0,
    "double_four": 12.0,
    "five": 18.0,
    "three_plus_one": 3.8,
    "two_plus_two": 4.8,
    "two_plus_one": 2.8,
    "double_three_plus_one": 10.5,
    "double_two_plus_two": 9.8,
    "double_two_plus_one": 8.2,
    "two_plus_one_three_plus_one": 9.0,
    "live_three_three_plus_one": 10.2,
    "live_three_two_plus_one": 8.7,
    "live_three_two_plus_two": 9.2,
    "two_plus_one_two_plus_two": 8.4,
}

DEFENSE_PENALTY_WEIGHTS: Dict[str, float] = {
    "live_three": 2.5,
    "sleep_three": 1.2,
    "live_four": 8.0,
    "rush_four": 6.0,
    "double_three": 8.0,
    "four_three": 12.0,
    "double_four": 15.0,
    "five": 22.0,
    "three_plus_one": 5.5,
    "two_plus_two": 6.0,
    "two_plus_one": 4.6,
    "double_three_plus_one": 13.0,
    "double_two_plus_two": 12.2,
    "double_two_plus_one": 10.5,
    "two_plus_one_three_plus_one": 11.6,
    "live_three_three_plus_one": 12.8,
    "live_three_two_plus_one": 11.0,
    "live_three_two_plus_two": 11.4,
    "two_plus_one_two_plus_two": 10.8,
}


def _count_occurrence(line: str, token: str) -> int:
    count = 0
    start = 0
    while True:
        idx = line.find(token, start)
        if idx < 0:
            break
        count += 1
        start = idx + 1
    return count


def _line_chars(board: np.ndarray, player: int, r: int, c: int, dr: int, dc: int, radius: int = 6) -> str:
    n = board.shape[0]
    chars = ["O"]
    for k in range(-radius, radius + 1):
        rr = r + dr * k
        cc = c + dc * k
        if rr < 0 or rr >= n or cc < 0 or cc >= n:
            chars.append("O")
            continue
        v = int(board[rr, cc])
        if v == player:
            chars.append("X")
        elif v == -player:
            chars.append("O")
        else:
            chars.append(".")
    chars.append("O")
    return "".join(chars)


def _scan_line(line: str) -> Dict[str, int]:
    live_three = _count_occurrence(line, ".XXX.") + _count_occurrence(line, ".XX.X.") + _count_occurrence(line, ".X.XX.")
    sleep_three = (
        _count_occurrence(line, "OXXX.")
        + _count_occurrence(line, ".XXXO")
        + _count_occurrence(line, "OXX.X.")
        + _count_occurrence(line, ".X.XXO")
    )
    live_four = _count_occurrence(line, ".XXXX.")
    rush_four = _count_occurrence(line, "OXXXX.") + _count_occurrence(line, ".XXXXO") + _count_occurrence(line, "XX.XX")
    five = _count_occurrence(line, "XXXXX")
    three_plus_one = _count_occurrence(line, "XXX.X") + _count_occurrence(line, "X.XXX")
    two_plus_two = _count_occurrence(line, "XX.XX")
    two_plus_one = _count_occurrence(line, "XX.X") + _count_occurrence(line, "X.XX")
    return {
        "live_three": live_three,
        "sleep_three": sleep_three,
        "live_four": live_four,
        "rush_four": rush_four,
        "five": five,
        "three_plus_one": three_plus_one,
        "two_plus_two": two_plus_two,
        "two_plus_one": two_plus_one,
    }


def local_pattern_features(board: np.ndarray, player: int, r: int, c: int) -> Dict[str, int]:
    merged = {
        "live_three": 0,
        "sleep_three": 0,
        "live_four": 0,
        "rush_four": 0,
        "double_three": 0,
        "four_three": 0,
        "double_four": 0,
        "five": 0,
        "three_plus_one": 0,
        "two_plus_two": 0,
        "two_plus_one": 0,
        "double_three_plus_one": 0,
        "double_two_plus_two": 0,
        "double_two_plus_one": 0,
        "two_plus_one_three_plus_one": 0,
        "live_three_three_plus_one": 0,
        "live_three_two_plus_one": 0,
        "live_three_two_plus_two": 0,
        "two_plus_one_two_plus_two": 0,
    }
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        line = _line_chars(board, player, r, c, dr, dc)
        cur = _scan_line(line)
        merged["live_three"] += cur["live_three"]
        merged["sleep_three"] += cur["sleep_three"]
        merged["live_four"] += cur["live_four"]
        merged["rush_four"] += cur["rush_four"]
        merged["five"] += cur["five"]
        merged["three_plus_one"] += cur["three_plus_one"]
        merged["two_plus_two"] += cur["two_plus_two"]
        merged["two_plus_one"] += cur["two_plus_one"]

    four_total = merged["live_four"] + merged["rush_four"]
    if merged["live_three"] >= 2:
        merged["double_three"] = 1
    if four_total >= 2:
        merged["double_four"] = 1
    if four_total >= 1 and merged["live_three"] >= 1:
        merged["four_three"] = 1
    if merged["five"] > 0:
        merged["five"] = 1

    if merged["three_plus_one"] >= 2:
        merged["double_three_plus_one"] = 1
    if merged["two_plus_two"] >= 2:
        merged["double_two_plus_two"] = 1
    if merged["two_plus_one"] >= 2:
        merged["double_two_plus_one"] = 1
    if merged["two_plus_one"] >= 1 and merged["three_plus_one"] >= 1:
        merged["two_plus_one_three_plus_one"] = 1
    if merged["live_three"] >= 1 and merged["three_plus_one"] >= 1:
        merged["live_three_three_plus_one"] = 1
    if merged["live_three"] >= 1 and merged["two_plus_one"] >= 1:
        merged["live_three_two_plus_one"] = 1
    if merged["live_three"] >= 1 and merged["two_plus_two"] >= 1:
        merged["live_three_two_plus_two"] = 1
    if merged["two_plus_one"] >= 1 and merged["two_plus_two"] >= 1:
        merged["two_plus_one_two_plus_two"] = 1
    return merged


def weighted_pattern_score(features: Dict[str, int], weights: Dict[str, float]) -> float:
    return float(sum(float(features.get(k, 0)) * float(w) for k, w in weights.items()))


def evaluate_motif_delta(
    board_before: np.ndarray,
    board_after: np.ndarray,
    player: int,
    r: int,
    c: int,
) -> tuple[float, float]:
    """
    Returns:
    - reward for creating own tactical motif
    - penalty for leaving opponent tactical motif unblocked
    """
    self_after = local_pattern_features(board_after, player, r, c)
    opp_before = local_pattern_features(board_before, -player, r, c)
    opp_after = local_pattern_features(board_after, -player, r, c)

    self_reward = weighted_pattern_score(self_after, ATTACK_WEIGHTS)
    opp_before_score = weighted_pattern_score(opp_before, DEFENSE_PENALTY_WEIGHTS)
    opp_after_score = weighted_pattern_score(opp_after, DEFENSE_PENALTY_WEIGHTS)

    # Blocking opponent threats should be rewarded implicitly.
    block_bonus = max(0.0, opp_before_score - opp_after_score) * 0.45
    missed_penalty = max(0.0, opp_after_score - 0.35 * opp_before_score)
    return self_reward + block_bonus, missed_penalty
