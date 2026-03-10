from __future__ import annotations

from itertools import combinations

import numpy as np

from gomoku_logic import BOARD_SIZE
from supervised.patterns import ATTACK_WEIGHTS, DEFENSE_PENALTY_WEIGHTS, local_pattern_features


def pick_lp_game_theory_move(board: np.ndarray, player: int, legal_moves: list[int]) -> tuple[int, np.ndarray]:
    legal = [int(a) for a in legal_moves]
    if not legal:
        raise ValueError("No legal move available")

    row_actions = _lp_candidate_actions(board, player, legal, max_candidates=8)
    col_actions = _lp_candidate_actions(board, -player, legal, max_candidates=8)
    if not row_actions:
        action = legal[0]
        return action, _one_hot(action)
    if not col_actions:
        action = row_actions[0]
        return action, _one_hot(action)

    payoff = _build_lp_payoff_matrix(board, player, row_actions, col_actions)
    mixed_row = _solve_zero_sum_lp_support_enum(payoff)
    if mixed_row is None or float(mixed_row.sum()) <= 1e-8:
        action = row_actions[0]
        return action, _one_hot(action)

    best_local = int(np.argmax(mixed_row))
    action = int(row_actions[best_local])
    policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    for i, a in enumerate(row_actions):
        policy[int(a)] = float(max(0.0, mixed_row[i]))
    s = float(policy.sum())
    if s > 1e-8:
        policy /= s
    else:
        policy = _one_hot(action)
    return action, policy


def _one_hot(action: int) -> np.ndarray:
    out = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    out[int(action)] = 1.0
    return out


def _local_motif_score(board: np.ndarray, player: int, r: int, c: int, weights: dict[str, float]) -> float:
    feats = local_pattern_features(board, int(player), int(r), int(c))
    return float(sum(float(feats.get(k, 0)) * float(w) for k, w in weights.items()))


def _lp_candidate_actions(board: np.ndarray, player: int, legal: list[int], max_candidates: int = 8) -> list[int]:
    if len(legal) <= max_candidates:
        return list(legal)
    center = (BOARD_SIZE - 1) / 2.0
    scored: list[tuple[float, int]] = []
    for a in legal:
        r, c = divmod(int(a), BOARD_SIZE)
        att = _local_motif_score(board, int(player), r, c, ATTACK_WEIGHTS)
        deff = _local_motif_score(board, int(-player), r, c, DEFENSE_PENALTY_WEIGHTS)
        center_bonus = -0.03 * (abs(r - center) + abs(c - center))
        score = 1.05 * att + 1.20 * deff + center_bonus
        scored.append((float(score), int(a)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [a for _, a in scored[: max(2, int(max_candidates))]]


def _build_lp_payoff_matrix(
    board: np.ndarray,
    player: int,
    row_actions: list[int],
    col_actions: list[int],
) -> np.ndarray:
    m = len(row_actions)
    n = len(col_actions)
    payoff = np.zeros((m, n), dtype=np.float64)
    player = int(player)
    opp = -player

    for i, a in enumerate(row_actions):
        board_after_ours = board.copy()
        ra, ca = divmod(int(a), BOARD_SIZE)
        board_after_ours[ra, ca] = player

        if _has_five_from(board_after_ours, player, ra, ca):
            payoff[i, :] = 1.0
            continue

        fallback_opp = _worst_case_opp_response(board_after_ours, player, opp, col_actions)
        for j, b in enumerate(col_actions):
            board_after_two = board_after_ours.copy()
            rb, cb = divmod(int(b), BOARD_SIZE)
            if board_after_two[rb, cb] == 0:
                chosen_opp = int(b)
            else:
                chosen_opp = int(fallback_opp) if fallback_opp is not None else -1
            if chosen_opp >= 0:
                r2, c2 = divmod(chosen_opp, BOARD_SIZE)
                board_after_two[r2, c2] = opp
                if _has_five_from(board_after_two, opp, r2, c2):
                    payoff[i, j] = -1.0
                    continue
            payoff[i, j] = _position_payoff(board_after_two, player, a, chosen_opp)
    return payoff


def _worst_case_opp_response(
    board_after_ours: np.ndarray,
    player: int,
    opp: int,
    col_actions: list[int],
) -> int | None:
    best_action = None
    best_val = None
    for b in col_actions:
        r, c = divmod(int(b), BOARD_SIZE)
        if board_after_ours[r, c] != 0:
            continue
        tmp = board_after_ours.copy()
        tmp[r, c] = int(opp)
        if _has_five_from(tmp, int(opp), r, c):
            return int(b)
        val = _position_payoff(tmp, int(player), None, int(b))
        if best_val is None or val < best_val:
            best_val = val
            best_action = int(b)
    return best_action


def _position_payoff(board: np.ndarray, player: int, my_action: int | None, opp_action: int | None) -> float:
    my_attack = 0.0
    opp_attack = 0.0
    if my_action is not None:
        r, c = divmod(int(my_action), BOARD_SIZE)
        my_attack += _local_motif_score(board, int(player), r, c, ATTACK_WEIGHTS)
        opp_attack += _local_motif_score(board, int(-player), r, c, ATTACK_WEIGHTS)
    if opp_action is not None and int(opp_action) >= 0:
        r, c = divmod(int(opp_action), BOARD_SIZE)
        my_attack += 0.25 * _local_motif_score(board, int(player), r, c, ATTACK_WEIGHTS)
        opp_attack += 0.85 * _local_motif_score(board, int(-player), r, c, ATTACK_WEIGHTS)
    raw = float(my_attack - opp_attack)
    return float(np.tanh(raw / 10.0))


def _solve_zero_sum_lp_support_enum(payoff: np.ndarray) -> np.ndarray | None:
    a = np.asarray(payoff, dtype=np.float64)
    if a.ndim != 2:
        return None
    m, n = a.shape
    if m == 0:
        return None
    if n == 0:
        return np.ones((m,), dtype=np.float64) / float(m)

    kmax = min(m, n)
    tol = 1e-7
    best_x = None
    best_v = -1e18

    rows = list(range(m))
    cols = list(range(n))
    for k in range(1, kmax + 1):
        for rset in combinations(rows, k):
            rlist = list(rset)
            for cset in combinations(cols, k):
                clist = list(cset)
                sub = a[np.ix_(rlist, clist)]

                lhs_x = np.zeros((k + 1, k + 1), dtype=np.float64)
                lhs_x[:k, :k] = sub.T
                lhs_x[:k, k] = -1.0
                lhs_x[k, :k] = 1.0
                rhs_x = np.zeros((k + 1,), dtype=np.float64)
                rhs_x[k] = 1.0
                try:
                    sol_x = np.linalg.solve(lhs_x, rhs_x)
                except np.linalg.LinAlgError:
                    continue
                x_sub = sol_x[:k]

                lhs_y = np.zeros((k + 1, k + 1), dtype=np.float64)
                lhs_y[:k, :k] = sub
                lhs_y[:k, k] = -1.0
                lhs_y[k, :k] = 1.0
                rhs_y = np.zeros((k + 1,), dtype=np.float64)
                rhs_y[k] = 1.0
                try:
                    sol_y = np.linalg.solve(lhs_y, rhs_y)
                except np.linalg.LinAlgError:
                    continue
                y_sub = sol_y[:k]

                if np.any(x_sub < -tol) or np.any(y_sub < -tol):
                    continue

                x = np.zeros((m,), dtype=np.float64)
                y = np.zeros((n,), dtype=np.float64)
                x[rlist] = np.maximum(0.0, x_sub)
                y[clist] = np.maximum(0.0, y_sub)
                sx = float(x.sum())
                sy = float(y.sum())
                if sx <= tol or sy <= tol:
                    continue
                x /= sx
                y /= sy

                col_values = x @ a
                row_values = a @ y
                lower = float(col_values.min())
                upper = float(row_values.max())
                if lower + 5e-5 < upper:
                    continue
                if lower > best_v:
                    best_v = lower
                    best_x = x

    if best_x is not None:
        return best_x

    mins = a.min(axis=1)
    idx = int(np.argmax(mins))
    out = np.zeros((m,), dtype=np.float64)
    out[idx] = 1.0
    return out


def _has_five_from(board: np.ndarray, player: int, r: int, c: int) -> bool:
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        count = 1
        count += _count_dir(board, player, r, c, dr, dc)
        count += _count_dir(board, player, r, c, -dr, -dc)
        if count >= 5:
            return True
    return False


def _count_dir(board: np.ndarray, player: int, r: int, c: int, dr: int, dc: int) -> int:
    nr, nc = r + dr, c + dc
    cnt = 0
    while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and int(board[nr, nc]) == int(player):
        cnt += 1
        nr += dr
        nc += dc
    return cnt
