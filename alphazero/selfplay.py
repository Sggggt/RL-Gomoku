from typing import List, Tuple, Dict, Any
import os
import time
import uuid
import numpy as np
import torch
from tqdm import trange, tqdm
import torch.multiprocessing as mp

from gomoku_logic import Gomoku, BOARD_SIZE
from .mcts import MCTS
from .network import GomokuNet
try:
    from supervised.patterns import evaluate_motif_delta
except Exception:
    evaluate_motif_delta = None


def _immediate_winning_actions(board: np.ndarray, player: int) -> list[int]:
    actions: list[int] = []
    empties = np.argwhere(board == 0)
    n = board.shape[0]
    for r, c in empties:
        rr = int(r)
        cc = int(c)
        if _is_winning_move(board, player, rr, cc, n):
            actions.append(rr * n + cc)
    return actions


def _is_winning_move(board: np.ndarray, player: int, r: int, c: int, n: int) -> bool:
    if board[r, c] != 0:
        return False
    board[r, c] = player
    try:
        return _has_five_from(board, player, r, c, n)
    finally:
        board[r, c] = 0


def _has_five_from(board: np.ndarray, player: int, r: int, c: int, n: int) -> bool:
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        count = 1
        count += _count_dir(board, player, r, c, dr, dc, n)
        count += _count_dir(board, player, r, c, -dr, -dc, n)
        if count >= 5:
            return True
    return False


def _count_dir(board: np.ndarray, player: int, r: int, c: int, dr: int, dc: int, n: int) -> int:
    nr, nc = r + dr, c + dc
    cnt = 0
    while 0 <= nr < n and 0 <= nc < n and int(board[nr, nc]) == player:
        cnt += 1
        nr += dr
        nc += dc
    return cnt


SELF_CHAIN_REWARD = {
    3: 1.8,
    4: 4.5,
    5: 12.0,
}

UNBLOCKED_OPP_PENALTY = {
    4: 7.0,
    5: 20.0,
}


def _max_line_from(board: np.ndarray, player: int, r: int, c: int, n: int) -> int:
    best = 1
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        count = 1
        count += _count_dir(board, player, r, c, dr, dc, n)
        count += _count_dir(board, player, r, c, -dr, -dc, n)
        if count > best:
            best = count
    return best


def _has_any_five(board: np.ndarray, player: int) -> bool:
    stones = np.argwhere(board == player)
    n = board.shape[0]
    for r, c in stones:
        if _has_five_from(board, player, int(r), int(c), n):
            return True
    return False


def _chain_reward(chain_len: int) -> float:
    if chain_len >= 5:
        return SELF_CHAIN_REWARD[5]
    if chain_len == 4:
        return SELF_CHAIN_REWARD[4]
    if chain_len == 3:
        return SELF_CHAIN_REWARD[3]
    return 0.0


def _threat_level(board: np.ndarray, player: int) -> int:
    if _has_any_five(board, player):
        return 5
    if _immediate_winning_actions(board, player):
        return 4
    return 0


def _normalize_actions(actions: list[int], size: int) -> np.ndarray:
    target = np.zeros(size, dtype=np.float32)
    if not actions:
        return target
    p = 1.0 / float(len(actions))
    for a in actions:
        target[a] = p
    return target


def _cell_symbol(board: np.ndarray, r: int, c: int, player: int) -> str:
    v = int(board[r, c])
    if v == int(player):
        return "X"
    if v == 0:
        return "."
    return "O"


def _collect_line(r0: int, c0: int, dr: int, dc: int, n: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    r, c = r0, c0
    while 0 <= r < n and 0 <= c < n:
        out.append((r, c))
        r += dr
        c += dc
    return out


def _all_lines(n: int) -> list[list[tuple[int, int]]]:
    lines: list[list[tuple[int, int]]] = []
    for r in range(n):
        lines.append([(r, c) for c in range(n)])
    for c in range(n):
        lines.append([(r, c) for r in range(n)])
    for c0 in range(n):
        lines.append(_collect_line(0, c0, 1, 1, n))
    for r0 in range(1, n):
        lines.append(_collect_line(r0, 0, 1, 1, n))
    for c0 in range(n):
        lines.append(_collect_line(0, c0, 1, -1, n))
    for r0 in range(1, n):
        lines.append(_collect_line(r0, n - 1, 1, -1, n))
    return [line for line in lines if len(line) >= 5]


def _open_three_actions_from_line(padded: str, coords: list[tuple[int, int]], n: int) -> set[int]:
    actions: set[int] = set()
    patterns = [
        (".XXX.", [0, 4]),
        (".XX.X.", [0, 3, 5]),
        (".X.XX.", [0, 2, 5]),
    ]
    for pat, empty_pos in patterns:
        start = 0
        while True:
            i = padded.find(pat, start)
            if i < 0:
                break
            # Skip pseudo-live-three shape: O.XXX.O (user notation: x.ooo.x).
            # Both candidate extension points are externally blocked by opponent/boundary.
            if pat == ".XXX.":
                left_outer = padded[i - 1] if i - 1 >= 0 else "O"
                right_outer_idx = i + len(pat)
                right_outer = padded[right_outer_idx] if right_outer_idx < len(padded) else "O"
                if left_outer == "O" and right_outer == "O":
                    start = i + 1
                    continue
            for pos in empty_pos:
                line_idx = i + pos - 1
                if 0 <= line_idx < len(coords):
                    r, c = coords[line_idx]
                    actions.add(r * n + c)
            start = i + 1
    return actions


def _open_three_actions(board: np.ndarray, player: int) -> list[int]:
    n = board.shape[0]
    actions: set[int] = set()
    for coords in _all_lines(n):
        symbols = "".join(_cell_symbol(board, r, c, player) for r, c in coords)
        padded = "O" + symbols + "O"
        actions.update(_open_three_actions_from_line(padded, coords, n))
    return sorted(actions)


def _max_line_len(board: np.ndarray, player: int, r: int, c: int) -> int:
    n = board.shape[0]
    best = 1
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        cnt = 1
        cnt += _count_dir(board, player, r, c, dr, dc, n)
        cnt += _count_dir(board, player, r, c, -dr, -dc, n)
        best = max(best, cnt)
    return best


def _jump_pattern_counts(board: np.ndarray, player: int, r: int, c: int) -> tuple[int, int]:
    """
    Count jump motifs that pass through (r, c):
    - 2+1: three stones with one internal gap in a 4-cell window (e.g. XX.X, X.XX)
    - 3+1: four stones with one internal gap in a 5-cell window (e.g. XXX.X, XX.XX, X.XXX)
    """
    n = board.shape[0]
    jump21 = 0
    jump31 = 0
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        # Collect full line symbols through (r, c).
        line: list[str] = []
        idx = 0
        rr, cc = r, c
        while 0 <= rr - dr < n and 0 <= cc - dc < n:
            rr -= dr
            cc -= dc
        while 0 <= rr < n and 0 <= cc < n:
            if rr == r and cc == c:
                idx = len(line)
            line.append(_cell_symbol(board, rr, cc, player))
            rr += dr
            cc += dc

        has21 = False
        has31 = False
        for win_len in (4, 5):
            start_lo = max(0, idx - win_len + 1)
            start_hi = min(idx, len(line) - win_len)
            for s in range(start_lo, start_hi + 1):
                seg = line[s : s + win_len]
                if "O" in seg:
                    continue
                dots = seg.count(".")
                if win_len == 4:
                    if seg.count("X") == 3 and dots == 1 and seg[0] != "." and seg[-1] != ".":
                        has21 = True
                else:
                    if seg.count("X") == 4 and dots == 1 and seg[0] != "." and seg[-1] != ".":
                        has31 = True
        if has21:
            jump21 += 1
        if has31:
            jump31 += 1
    return jump21, jump31


def _offense_four_actions(board: np.ndarray, player: int) -> list[int]:
    n = board.shape[0]
    acts: list[int] = []
    empties = np.argwhere(board == 0)
    for rr, cc in empties:
        r = int(rr)
        c = int(cc)
        board[r, c] = int(player)
        try:
            if _max_line_len(board, int(player), r, c) >= 4:
                acts.append(r * n + c)
        finally:
            board[r, c] = 0
    return acts


def _sample_from_candidates(policy: np.ndarray, candidates: list[int]) -> int:
    if not candidates:
        return int(np.random.choice(len(policy), p=policy))
    cand = np.array(sorted(set(int(a) for a in candidates)), dtype=np.int64)
    probs = np.asarray(policy[cand], dtype=np.float64)
    s = float(probs.sum())
    if not np.isfinite(s) or s <= 0.0:
        probs = np.ones(len(cand), dtype=np.float64) / float(len(cand))
    else:
        probs = probs / s
    return int(np.random.choice(cand, p=probs))


def _guardrail_select_action(game: Gomoku, policy: np.ndarray) -> tuple[int, np.ndarray, float]:
    legal = set(game.legal_moves())
    player = int(game.player)
    opp = -player
    size = int(policy.size)
    guard = np.zeros(size, dtype=np.float32)
    guard_weight = 0.0
    model_pref = int(np.argmax(policy))

    win = [a for a in _immediate_winning_actions(game.board, player) if a in legal]
    if win:
        action = _sample_from_candidates(policy, win)
        if model_pref not in set(win):
            guard = _normalize_actions(win, size)
            guard_weight = 1.0
        return action, guard, guard_weight

    block = [a for a in _immediate_winning_actions(game.board, opp) if a in legal]
    if block:
        action = _sample_from_candidates(policy, block)
        if model_pref not in set(block):
            guard = _normalize_actions(block, size)
            guard_weight = 1.0
        return action, guard, guard_weight

    own_open_three = [a for a in _open_three_actions(game.board, player) if a in legal]
    if own_open_three:
        action = _sample_from_candidates(policy, own_open_three)
        if model_pref not in set(own_open_three):
            guard = _normalize_actions(own_open_three, size)
            guard_weight = 1.0
        return action, guard, guard_weight

    opp_open_three = [a for a in _open_three_actions(game.board, opp) if a in legal]
    own_four = [a for a in _offense_four_actions(game.board, player) if a in legal]
    if opp_open_three or own_four:
        cands = sorted(set(opp_open_three).union(own_four))
        action = _sample_from_candidates(policy, cands)
        if model_pref not in set(cands):
            guard = _normalize_actions(cands, size)
            guard_weight = 1.0
        return action, guard, guard_weight

    return int(np.random.choice(size, p=policy)), guard, guard_weight


def _survival_prior_target(game: Gomoku, player: int) -> np.ndarray:
    legal = game.legal_moves()
    size = game.board.size
    target = np.zeros(size, dtype=np.float32)
    if not legal:
        return target

    instant_wins = set(_immediate_winning_actions(game.board, player))
    n = game.board.shape[0]
    opp_threat_before = _threat_level(game.board, -player)
    scores = []
    for action in legal:
        rr, cc = divmod(int(action), n)
        g2 = game.clone()
        try:
            g2.step(action)
        except ValueError:
            scores.append(-1e9)
            continue
        if g2.finished and g2.winner == player:
            scores.append(100.0)
            continue

        opp_immediate = len(_immediate_winning_actions(g2.board, -player))
        self_immediate = len(_immediate_winning_actions(g2.board, player))
        self_chain = _max_line_from(g2.board, player, rr, cc, n)
        self_chain_reward = _chain_reward(self_chain)
        jump21, jump31 = _jump_pattern_counts(g2.board, player, rr, cc)
        jump_reward = 0.90 * float(jump21) + 1.60 * float(jump31)

        opp_threat_after = _threat_level(g2.board, -player)
        missed_block_penalty = 0.0
        if opp_threat_before >= 5 and opp_threat_after >= 5:
            missed_block_penalty = UNBLOCKED_OPP_PENALTY[5]
        elif opp_threat_before >= 4 and opp_threat_after >= 4:
            missed_block_penalty = UNBLOCKED_OPP_PENALTY[4]
        if opp_threat_after >= 5:
            missed_block_penalty = max(missed_block_penalty, UNBLOCKED_OPP_PENALTY[5])

        # Prefer direct connect/defense in high-urgency states, otherwise encourage jump motifs.
        if self_chain >= 4 or self_immediate > 0:
            jump_reward *= 0.35
        if opp_immediate > 0 or opp_threat_before >= 4:
            jump_reward *= 0.30

        quick_value = 1.2 * float(self_immediate) + self_chain_reward
        quick_value += jump_reward
        if action in instant_wins:
            quick_value += SELF_CHAIN_REWARD[5]
        if evaluate_motif_delta is not None:
            motif_reward, motif_penalty = evaluate_motif_delta(game.board, g2.board, player, rr, cc)
            quick_value += 0.45 * float(motif_reward)
        else:
            motif_penalty = 0.0
        defense_penalty = 1.2 * float(opp_immediate) + missed_block_penalty
        defense_penalty += 0.65 * float(motif_penalty)
        score = quick_value - defense_penalty
        scores.append(score)

    score_arr = np.array(scores, dtype=np.float32)
    score_arr = score_arr - float(score_arr.max())
    probs = np.exp(score_arr)
    probs_sum = float(probs.sum())
    if probs_sum <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / probs_sum
    for idx, action in enumerate(legal):
        target[action] = float(probs[idx])
    return target


def _phase_target(offense_actions: list[int], defense_actions: list[int], move_count: int) -> float:
    offense_n = float(len(offense_actions))
    defense_n = float(len(defense_actions))
    if offense_n > 0 and defense_n == 0:
        return 0.85
    if defense_n > 0 and offense_n == 0:
        return 0.15
    if offense_n > 0 and defense_n > 0:
        return 0.25
    # Opening tends to be shape-building and cautious.
    if move_count < 8:
        return 0.4
    return 0.55


def _safe_policy(policy: np.ndarray) -> np.ndarray:
    p = np.asarray(policy, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        return np.ones(BOARD_SIZE * BOARD_SIZE, dtype=np.float32) / float(BOARD_SIZE * BOARD_SIZE)
    if not np.all(np.isfinite(p)):
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.maximum(p, 0.0)
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0.0:
        return np.ones_like(p, dtype=np.float32) / float(p.size)
    return (p / s).astype(np.float32, copy=False)


def _heuristic_action(game: Gomoku, player: int) -> int:
    legal = game.legal_moves()
    if not legal:
        raise ValueError("No legal moves for heuristic action.")
    n = game.board.shape[0]

    win_actions = _immediate_winning_actions(game.board, player)
    if win_actions:
        return int(np.random.choice(win_actions))

    block_actions = _immediate_winning_actions(game.board, -player)
    if block_actions:
        return int(np.random.choice(block_actions))

    stones = np.argwhere(game.board != 0)
    center = (n - 1) / 2.0
    best_score = -1e9
    best_actions: list[int] = []
    for action in legal:
        r, c = divmod(int(action), n)
        center_score = -0.06 * (abs(r - center) + abs(c - center))
        if len(stones) == 0:
            prox_score = 0.0
        else:
            dmin = float(np.min(np.abs(stones[:, 0] - r) + np.abs(stones[:, 1] - c)))
            prox_score = -0.18 * dmin
        own_line = 0
        opp_line = 0
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            own_line = max(
                own_line,
                _count_dir(game.board, player, r, c, dr, dc, n)
                + _count_dir(game.board, player, r, c, -dr, -dc, n),
            )
            opp_line = max(
                opp_line,
                _count_dir(game.board, -player, r, c, dr, dc, n)
                + _count_dir(game.board, -player, r, c, -dr, -dc, n),
            )
        shape_score = 0.34 * float(own_line) + 0.22 * float(opp_line)
        score = center_score + prox_score + shape_score
        if score > best_score:
            best_score = score
            best_actions = [int(action)]
        elif score == best_score:
            best_actions.append(int(action))
    return int(np.random.choice(best_actions))


def self_play_games(
    net,
    num_games: int,
    device: str = "cpu",
    simulations: int = 200,
    temp_threshold: int = 20,
    show_progress: bool = True,
    evaluator=None,
    posterior_stats: dict | None = None,
    posterior_scale: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Runs self-play to generate training data.
    Returns list of dict samples with:
    state, policy, value, offense, defense, hardness.
    """
    if net:
        net.eval()
    data = []
    iterator = trange(num_games, desc="Self-play games", ncols=80) if show_progress else range(num_games)
    for _ in iterator:
        game = Gomoku()
        mcts = MCTS(
            net,
            device=device,
            simulations=simulations,
            evaluator=evaluator,
            posterior_stats=posterior_stats,
            posterior_scale=posterior_scale,
        )
        history = []
        move_count = 0
        while not game.finished:
            temperature = 0.9 if move_count < temp_threshold else 0.05
            policy = _safe_policy(mcts.run(game, add_noise=False, temperature=temperature))
            action, guard_target, guard_weight = _guardrail_select_action(game, policy)
            if float(guard_weight) > 0.0 and float(np.sum(guard_target)) > 0.0:
                policy_target = guard_target.astype(np.float32, copy=True)
            else:
                policy_target = policy.astype(np.float32, copy=True)
            state_planes = game.canonical_board()
            offense_actions = _immediate_winning_actions(game.board, game.player)
            defense_actions = _immediate_winning_actions(game.board, -game.player)
            offense = _normalize_actions(offense_actions, len(policy))
            defense = _normalize_actions(defense_actions, len(policy))
            is_hard = bool(len(defense_actions) > 0 or len(offense_actions) > 0)
            prior_target = _survival_prior_target(game, game.player)
            phase_target = _phase_target(offense_actions, defense_actions, move_count)
            history.append(
                (
                    state_planes,
                    policy_target,
                    game.player,
                    offense,
                    defense,
                    is_hard,
                    prior_target,
                    phase_target,
                    guard_target,
                    guard_weight,
                )
            )
            game.step(action)
            move_count += 1

        # assign value to states from each player's perspective
        for (
            state_planes,
            policy,
            player,
            offense,
            defense,
            is_hard,
            prior_target,
            phase_target,
            guard_target,
            guard_weight,
        ) in history:
            if game.winner == 0:
                value = 0.0
            else:
                value = 1.0 if game.winner == player else -1.0
            data.append(
                {
                    "state": state_planes.astype(np.float32, copy=True),
                    "policy": policy.astype(np.float32, copy=True),
                    "value": float(value),
                    "offense": offense.astype(np.float32, copy=True),
                    "defense": defense.astype(np.float32, copy=True),
                    "guard": guard_target.astype(np.float32, copy=True),
                    "guard_weight": float(guard_weight),
                    "hardness": (3.0 if is_hard else 1.0) + (1.0 if float(guard_weight) > 0 else 0.0),
                    "prior_target": prior_target.astype(np.float32, copy=True),
                    "phase_target": float(phase_target),
                }
            )
    return data


def self_play_games_vs_heuristic(
    net,
    num_games: int,
    device: str = "cpu",
    simulations: int = 200,
    temp_threshold: int = 20,
    show_progress: bool = True,
    posterior_stats: dict | None = None,
    posterior_scale: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Generate data from net-vs-heuristic games.
    Only net moves are recorded as training samples.
    """
    if net:
        net.eval()
    data = []
    iterator = trange(num_games, desc="Heuristic games", ncols=80) if show_progress else range(num_games)
    for i in iterator:
        game = Gomoku()
        net_player = 1 if (i % 2 == 0) else -1
        mcts = MCTS(
            net,
            device=device,
            simulations=simulations,
            posterior_stats=posterior_stats,
            posterior_scale=posterior_scale,
        )
        history = []
        move_count = 0
        while not game.finished:
            if game.player == net_player:
                temperature = 0.9 if move_count < temp_threshold else 0.05
                policy = _safe_policy(mcts.run(game, add_noise=False, temperature=temperature))
                action, guard_target, guard_weight = _guardrail_select_action(game, policy)
                if float(guard_weight) > 0.0 and float(np.sum(guard_target)) > 0.0:
                    policy_target = guard_target.astype(np.float32, copy=True)
                else:
                    policy_target = policy.astype(np.float32, copy=True)
                state_planes = game.canonical_board()
                offense_actions = _immediate_winning_actions(game.board, game.player)
                defense_actions = _immediate_winning_actions(game.board, -game.player)
                offense = _normalize_actions(offense_actions, len(policy))
                defense = _normalize_actions(defense_actions, len(policy))
                is_hard = bool(len(defense_actions) > 0 or len(offense_actions) > 0)
                prior_target = _survival_prior_target(game, game.player)
                phase_target = _phase_target(offense_actions, defense_actions, move_count)
                history.append(
                    (
                        state_planes,
                        policy_target,
                        game.player,
                        offense,
                        defense,
                        is_hard,
                        prior_target,
                        phase_target,
                        guard_target,
                        guard_weight,
                    )
                )
            else:
                action = _heuristic_action(game, game.player)
            game.step(action)
            move_count += 1

        for (
            state_planes,
            policy,
            player,
            offense,
            defense,
            is_hard,
            prior_target,
            phase_target,
            guard_target,
            guard_weight,
        ) in history:
            if game.winner == 0:
                value = 0.0
            else:
                value = 1.0 if game.winner == player else -1.0
            data.append(
                {
                    "state": state_planes.astype(np.float32, copy=True),
                    "policy": policy.astype(np.float32, copy=True),
                    "value": float(value),
                    "offense": offense.astype(np.float32, copy=True),
                    "defense": defense.astype(np.float32, copy=True),
                    "guard": guard_target.astype(np.float32, copy=True),
                    "guard_weight": float(guard_weight),
                    "hardness": (3.0 if is_hard else 1.0) + (1.0 if float(guard_weight) > 0 else 0.0),
                    "prior_target": prior_target.astype(np.float32, copy=True),
                    "phase_target": float(phase_target),
                }
            )
    return data


def _worker_selfplay(args):
    (
        games,
        state_dict,
        device,
        simulations,
        temp_threshold,
        queues,
        posterior_stats,
        posterior_scale,
    ) = args
    evaluator = None
    net_device = device
    if queues is not None:
        # Use shared inference server; this worker can keep its model on CPU.
        net_device = "cpu"
        req_q, res_q, max_batch, timeout = queues
        evaluator = AsyncEvaluator(req_q, res_q, max_batch=max_batch, timeout=timeout)
    net = GomokuNet().to(net_device)
    net.load_state_dict(state_dict)

    return self_play_games(
        net,
        num_games=games,
        device=device,
        simulations=simulations,
        temp_threshold=temp_threshold,
        show_progress=False,
        evaluator=evaluator,
        posterior_stats=posterior_stats,
        posterior_scale=posterior_scale,
    )


def self_play_games_parallel(
    net,
    num_games: int,
    num_workers: int | None = None,
    device: str = "cpu",
    simulations: int = 200,
    temp_threshold: int = 20,
    batch_eval: bool = True,
    max_batch_size: int = 64,
    eval_timeout: float = 0.01,
    posterior_stats: dict | None = None,
    posterior_scale: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Parallel self-play using multiple processes.
    Optional batched inference server (shared GPU) to reduce per-state forward overhead.
    """
    if num_games <= 0:
        return []
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)
    num_workers = max(1, num_workers)

    # Distribute games across workers
    base = num_games // num_workers
    remainder = num_games % num_workers
    assignments = [base + (1 if i < remainder else 0) for i in range(num_workers)]
    assignments = [a for a in assignments if a > 0]
    if not assignments:
        return []

    state_dict = {k: v.cpu() for k, v in net.state_dict().items()}  # move to CPU for safer pickling
    ctx = mp.get_context("spawn")

    # Set up optional shared inference worker
    queues = None
    gpu_proc = None
    res_queues = None
    if batch_eval:
        if os.name == "nt":
            # Windows multiprocessing + Manager queues can hang on KeyboardInterrupt;
            # disable batch_eval by default on Windows unless explicitly requested.
            print("Batch eval disabled on Windows to avoid Manager queue hangs. Set batch_eval=True explicitly to force.")
        else:
            manager = ctx.Manager()
            req_q = manager.Queue(maxsize=4096)
            res_queues = [manager.Queue(maxsize=4096) for _ in assignments]
            gpu_proc = ctx.Process(
                target=_inference_worker,
                args=(state_dict, device, req_q, max_batch_size, eval_timeout),
                daemon=True,
            )
            gpu_proc.start()
            queues = (req_q, max_batch_size, eval_timeout)

    args = [
        (
            games,
            state_dict,
            device,
            simulations,
            temp_threshold,
            None if queues is None else (queues[0], res_queues[i], queues[1], queues[2]),
            posterior_stats,
            posterior_scale,
        )
        for i, games in enumerate(assignments)
    ]

    collected = []
    try:
        with ctx.Pool(len(args)) as pool, tqdm(
            total=len(args), desc="Worker batches", ncols=80
        ) as pbar:
            for res in pool.imap_unordered(_worker_selfplay, args):
                collected.append(res)
                pbar.update(1)
    finally:
        if gpu_proc is not None:
            gpu_proc.terminate()
            gpu_proc.join()

    data = [sample for part in collected for sample in part]
    return data


class AsyncEvaluator:
    """
    Lightweight client that queues single-state inference to a shared GPU worker,
    with small timeout-based micro-batching.
    """

    def __init__(self, req_q, res_q, max_batch: int = 64, timeout: float = 0.01):
        self.req_q = req_q
        self.res_q = res_q
        self.timeout = timeout
        self.max_batch = max_batch
        self.prefix = uuid.uuid4().hex  # avoid collisions across processes
        self.counter = 0

    def __call__(self, board_np: np.ndarray):
        req_id = f"{self.prefix}-{self.counter}"
        self.counter += 1
        self.req_q.put((req_id, board_np, self.res_q), block=True)
        while True:
            payload = self.res_q.get()
            if len(payload) == 7:
                rid, policy, value, prior, attack, defense, phase = payload
            elif len(payload) == 4:
                rid, policy, value, prior = payload
                attack = None
                defense = None
                phase = None
            else:
                rid, policy, value = payload
                prior = None
                attack = None
                defense = None
                phase = None
            if rid == req_id:
                if attack is not None and defense is not None and phase is not None:
                    return policy, value, prior, attack, defense, phase
                if prior is None:
                    return policy, value
                return policy, value, prior


def _inference_worker(state_dict, device, req_q, max_batch_size: int, timeout: float):
    """
    Dedicated process that batches inference requests from multiple self-play workers.
    """
    torch.set_num_threads(1)
    if device.startswith("cuda"):
        torch.cuda.set_device(0)
    net = GomokuNet().to(device)
    net.load_state_dict(state_dict)
    net.eval()
    pending = []
    last_flush = time.time()
    with torch.no_grad():
        while True:
            try:
                item = req_q.get(timeout=timeout)
                pending.append(item)
            except Exception:
                pass
            now = time.time()
            if pending and (len(pending) >= max_batch_size or now - last_flush >= timeout):
                ids, boards, reply_queues = zip(*pending)
                batch = torch.from_numpy(np.stack(boards)).float().to(device)
                outputs = net(batch)
                if isinstance(outputs, tuple) and len(outputs) == 6:
                    policy_logits, values, prior_logits, attack_logits, defense_logits, phase_logits = outputs
                    prior_logits = prior_logits.cpu().numpy()
                    attack_logits = attack_logits.cpu().numpy()
                    defense_logits = defense_logits.cpu().numpy()
                    phase_vals = torch.sigmoid(phase_logits).cpu().numpy()
                elif isinstance(outputs, tuple) and len(outputs) == 3:
                    policy_logits, values, prior_logits = outputs
                    attack_logits = None
                    defense_logits = None
                    phase_vals = None
                    prior_logits = prior_logits.cpu().numpy()
                else:
                    policy_logits, values = outputs
                    prior_logits = None
                    attack_logits = None
                    defense_logits = None
                    phase_vals = None
                policy_logits = policy_logits.cpu().numpy()
                values = values.cpu().numpy()
                if prior_logits is None:
                    for rid, p, v, reply_q in zip(ids, policy_logits, values, reply_queues):
                        reply_q.put((rid, p, float(v)))
                else:
                    if attack_logits is None or defense_logits is None or phase_vals is None:
                        for rid, p, v, pr, reply_q in zip(ids, policy_logits, values, prior_logits, reply_queues):
                            reply_q.put((rid, p, float(v), pr))
                    else:
                        for rid, p, v, pr, at, de, ph, reply_q in zip(
                            ids,
                            policy_logits,
                            values,
                            prior_logits,
                            attack_logits,
                            defense_logits,
                            phase_vals,
                            reply_queues,
                        ):
                            reply_q.put((rid, p, float(v), pr, at, de, float(ph)))
                pending = []
                last_flush = now
