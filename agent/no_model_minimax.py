from __future__ import annotations

from typing import Iterable

import numpy as np

from gomoku_logic import BOARD_SIZE
from supervised.patterns import ATTACK_WEIGHTS, DEFENSE_PENALTY_WEIGHTS, local_pattern_features


def pick_minimax_alpha_beta_move(
    board: np.ndarray,
    player: int,
    legal_moves: Iterable[int],
    depth: int = 2,
    max_branch: int = 10,
) -> tuple[int, np.ndarray]:
    legal = [int(a) for a in legal_moves]
    if not legal:
        raise ValueError("No legal move available")
    if len(legal) == 1:
        return legal[0], _one_hot(legal[0])

    board_work = board.copy()
    root = int(player)
    candidates = _rank_candidates(board_work, root, legal, max_branch=max_branch)
    if not candidates:
        return legal[0], _one_hot(legal[0])

    scores: list[float] = []
    best_action = candidates[0]
    best_score = -1e18
    alpha = -1e18
    beta = 1e18
    for a in candidates:
        r, c = divmod(int(a), BOARD_SIZE)
        board_work[r, c] = root
        if _has_five_from(board_work, root, r, c):
            score = 1.0
        else:
            score = _search(
                board=board_work,
                root_player=root,
                to_move=-root,
                depth=max(0, int(depth) - 1),
                alpha=alpha,
                beta=beta,
                last_action=int(a),
                player_just_moved=root,
                max_branch=max_branch,
            )
        board_work[r, c] = 0
        scores.append(float(score))
        if score > best_score:
            best_score = float(score)
            best_action = int(a)
        if best_score > alpha:
            alpha = best_score

    probs = _softmax(np.array(scores, dtype=np.float64), tau=0.35)
    policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    for i, a in enumerate(candidates):
        policy[int(a)] = float(probs[i])
    return int(best_action), policy


def _search(
    board: np.ndarray,
    root_player: int,
    to_move: int,
    depth: int,
    alpha: float,
    beta: float,
    last_action: int | None,
    player_just_moved: int,
    max_branch: int,
) -> float:
    if last_action is not None:
        r, c = divmod(int(last_action), BOARD_SIZE)
        if _has_five_from(board, int(player_just_moved), r, c):
            return 1.0 if int(player_just_moved) == int(root_player) else -1.0

    legal = _legal_moves(board)
    if not legal:
        return 0.0
    if depth <= 0:
        return _evaluate_board(board, root_player)

    moves = _rank_candidates(board, to_move, legal, max_branch=max_branch)
    if not moves:
        return _evaluate_board(board, root_player)

    maximizing = int(to_move) == int(root_player)
    if maximizing:
        best = -1e18
        for a in moves:
            r, c = divmod(int(a), BOARD_SIZE)
            board[r, c] = int(to_move)
            val = _search(
                board=board,
                root_player=root_player,
                to_move=-int(to_move),
                depth=depth - 1,
                alpha=alpha,
                beta=beta,
                last_action=int(a),
                player_just_moved=int(to_move),
                max_branch=max_branch,
            )
            board[r, c] = 0
            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return float(best)

    best = 1e18
    for a in moves:
        r, c = divmod(int(a), BOARD_SIZE)
        board[r, c] = int(to_move)
        val = _search(
            board=board,
            root_player=root_player,
            to_move=-int(to_move),
            depth=depth - 1,
            alpha=alpha,
            beta=beta,
            last_action=int(a),
            player_just_moved=int(to_move),
            max_branch=max_branch,
        )
        board[r, c] = 0
        if val < best:
            best = val
        if best < beta:
            beta = best
        if alpha >= beta:
            break
    return float(best)


def _evaluate_board(board: np.ndarray, root_player: int) -> float:
    legal = _legal_moves(board)
    if not legal:
        return 0.0
    sample = _rank_candidates(board, root_player, legal, max_branch=10)
    if not sample:
        sample = legal[: min(10, len(legal))]
    best_self = -1e18
    best_opp = -1e18
    for a in sample:
        r, c = divmod(int(a), BOARD_SIZE)
        self_att = _local_motif_score(board, root_player, r, c, ATTACK_WEIGHTS)
        opp_att = _local_motif_score(board, -root_player, r, c, ATTACK_WEIGHTS)
        opp_threat = _local_motif_score(board, -root_player, r, c, DEFENSE_PENALTY_WEIGHTS)
        best_self = max(best_self, self_att + 0.35 * opp_threat)
        best_opp = max(best_opp, opp_att)
    raw = float(best_self - 1.15 * best_opp)
    return float(np.tanh(raw / 12.0))


def _rank_candidates(board: np.ndarray, player: int, legal: list[int], max_branch: int) -> list[int]:
    if not legal:
        return []
    near = _neighbor_candidates(board, legal, radius=2)
    if near:
        base = near
    else:
        base = legal
    if len(base) <= max_branch:
        return list(base)

    center = (BOARD_SIZE - 1) / 2.0
    scored: list[tuple[float, int]] = []
    for a in base:
        r, c = divmod(int(a), BOARD_SIZE)
        att = _local_motif_score(board, int(player), r, c, ATTACK_WEIGHTS)
        deff = _local_motif_score(board, int(-player), r, c, DEFENSE_PENALTY_WEIGHTS)
        center_bonus = -0.03 * (abs(r - center) + abs(c - center))
        score = 1.00 * att + 1.15 * deff + center_bonus
        scored.append((float(score), int(a)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [a for _, a in scored[: max(2, int(max_branch))]]


def _neighbor_candidates(board: np.ndarray, legal: list[int], radius: int = 2) -> list[int]:
    stones = np.argwhere(board != 0)
    if len(stones) == 0:
        center = (BOARD_SIZE // 2) * BOARD_SIZE + (BOARD_SIZE // 2)
        return [center] if center in set(legal) else legal[:1]
    legal_set = set(int(a) for a in legal)
    out: set[int] = set()
    for rr, cc in stones:
        r = int(rr)
        c = int(cc)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr = r + dr
                nc = c + dc
                if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
                    continue
                a = nr * BOARD_SIZE + nc
                if a in legal_set:
                    out.add(a)
    return sorted(out)


def _legal_moves(board: np.ndarray) -> list[int]:
    empties = np.argwhere(board == 0)
    return [int(r) * BOARD_SIZE + int(c) for r, c in empties]


def _local_motif_score(board: np.ndarray, player: int, r: int, c: int, weights: dict[str, float]) -> float:
    feats = local_pattern_features(board, int(player), int(r), int(c))
    return float(sum(float(feats.get(k, 0)) * float(w) for k, w in weights.items()))


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


def _one_hot(action: int) -> np.ndarray:
    out = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
    out[int(action)] = 1.0
    return out


def _softmax(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) / max(1e-6, float(tau))
    z = z - np.max(z)
    e = np.exp(z)
    s = float(e.sum())
    if s <= 1e-12:
        return np.ones_like(e) / float(len(e))
    return e / s
