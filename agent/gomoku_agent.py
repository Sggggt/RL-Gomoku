from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from gomoku_logic import BOARD_SIZE, Gomoku
from alphazero import GomokuNet, MCTS
from supervised.patterns import ATTACK_WEIGHTS, DEFENSE_PENALTY_WEIGHTS, local_pattern_features
from .no_model_heuristic import center_prefer_move, pick_heuristic_move
from .no_model_lp import pick_lp_game_theory_move
from .no_model_minimax import pick_minimax_alpha_beta_move


@dataclass
class ModelChoice:
    path: Path | None
    label: str
    kind: str


@dataclass
class EpisodeStep:
    sample: dict
    actor_player: int


class GomokuAgent:
    """
    Minimal-intervention agent:
    1) Safety guardrail only for immediate win / immediate block.
    2) Otherwise decisions come from model + MCTS (or fallback if no model).
    Every recorded step is trained online with shaped RL reward.
    """

    def __init__(
        self,
        root: Path,
        device: str = "cpu",
        simulations: int = 80,
        online_lr: float = 1e-4,
        online_batch_size: int = 32,
        online_updates_per_step: int = 1,
        online_updates_endgame: int = 4,
        replay_capacity: int = 12000,
    ):
        self.root = root
        self.device = device
        self.simulations = simulations
        self.choices = self._discover_model_choices()
        self.choice_index = self._default_choice_index()
        self.net: GomokuNet | None = None
        self.model_error: str | None = None
        self.training_status: str | None = None
        self.posterior_stats: dict | None = None
        self.online_lr = online_lr
        self.online_batch_size = online_batch_size
        self.online_updates_per_step = online_updates_per_step
        self.online_updates_endgame = online_updates_endgame
        self.replay_capacity = replay_capacity
        self.online_optimizer: optim.Optimizer | None = None
        self.replay_buffer: deque[dict] = deque(maxlen=replay_capacity)
        self._episode_steps: list[EpisodeStep] = []
        self._pending_guardrail_target: np.ndarray | None = None
        self._pending_guardrail_weight: float = 0.0
        self._load_current_model()

    def _discover_model_choices(self) -> list[ModelChoice]:
        models_dir = self.root / "models"
        model_paths = sorted(models_dir.glob("*.pt")) if models_dir.exists() else []
        choices = [
            ModelChoice(path=None, label="Heuristic (No Model)", kind="heuristic"),
            ModelChoice(path=None, label="GameTheory LP", kind="lp"),
            ModelChoice(path=None, label="Minimax AlphaBeta", kind="minimax"),
        ]
        for path in model_paths:
            choices.append(ModelChoice(path=path, label=path.name, kind="model"))
        return choices

    def _default_choice_index(self) -> int:
        for i, choice in enumerate(self.choices):
            if choice.kind == "model" and choice.path is not None:
                return i
        return 0

    @property
    def current_model_label(self) -> str:
        return self.choices[self.choice_index].label

    def cycle_model(self) -> str | None:
        self.choice_index = (self.choice_index + 1) % len(self.choices)
        return self._load_current_model()

    def _load_current_model(self) -> str | None:
        choice = self.choices[self.choice_index]
        self.net = None
        self.model_error = None
        self.training_status = None
        self.posterior_stats = None
        if choice.kind != "model" or choice.path is None:
            return None
        try:
            net = GomokuNet().to(self.device)
            state = torch.load(choice.path, map_location=self.device)
            net.load_state_dict(state, strict=False)
            net.eval()
            self.net = net
            posterior_path = choice.path.with_name(choice.path.stem + "_posterior.pt")
            if posterior_path.exists():
                try:
                    self.posterior_stats = torch.load(posterior_path, map_location="cpu")
                except Exception:
                    self.posterior_stats = None
            self.online_optimizer = optim.Adam(self.net.parameters(), lr=self.online_lr, weight_decay=1e-4)
            return None
        except Exception as exc:
            self.model_error = f"加载模型失败 {choice.path.name}: {exc}"
            return self.model_error

    def pick_move(self, game: Gomoku) -> int:
        action, _ = self.pick_move_with_policy(game)
        return action

    def pick_move_with_policy(self, game: Gomoku) -> tuple[int, np.ndarray]:
        legal = game.legal_moves()
        if not legal:
            raise ValueError("No legal move available")
        legal_set = set(legal)
        self._pending_guardrail_target = None
        self._pending_guardrail_weight = 0.0

        # Guardrail only: immediate win / immediate block.
        immediate_win = [a for a in self._immediate_winning_actions(game.board, game.player) if a in legal_set]
        if immediate_win:
            action = self._rank_actions(game, immediate_win)
            return action, self._one_hot_policy(action)

        immediate_block = [a for a in self._immediate_winning_actions(game.board, -game.player) if a in legal_set]
        if immediate_block:
            action = self._rank_actions(game, immediate_block)
            return action, self._one_hot_policy(action)

        # Guardrail extension:
        # - block opponent live-three (including 2+1 shapes)
        # - connect own live-three (including 2+1 shapes)
        # - include own connect-4 opportunities
        own_open_three = [a for a in self._open_three_block_actions(game.board, game.player) if a in legal_set]
        if own_open_three:
            candidates = sorted(set(own_open_three))
            model_pref = self._model_preferred_action(game)
            if model_pref is not None and model_pref not in set(candidates):
                self._pending_guardrail_target = self._normalized_target(candidates)
                self._pending_guardrail_weight = 1.0
            action = self._rank_actions(game, candidates)
            return action, self._one_hot_policy(action)

        open_three_blocks = [a for a in self._open_three_block_actions(game.board, -game.player) if a in legal_set]
        own_four = [a for a in self._offense_four_actions(game.board, game.player) if a in legal_set]
        if open_three_blocks or own_four:
            candidates = sorted(set(open_three_blocks).union(own_four))
            model_pref = self._model_preferred_action(game)
            if model_pref is not None and model_pref not in set(candidates):
                self._pending_guardrail_target = self._normalized_target(candidates)
                self._pending_guardrail_weight = 1.0
            action = self._rank_actions(game, candidates)
            return action, self._one_hot_policy(action)

        if self.net is not None:
            mcts = MCTS(
                self.net,
                device=self.device,
                simulations=self.simulations,
                posterior_stats=self.posterior_stats,
                posterior_scale=0.5,
            )
            policy = mcts.run(game, add_noise=False, temperature=0.0)
            return int(policy.argmax()), policy.astype(np.float32)

        kind = self.choices[self.choice_index].kind
        if kind == "lp":
            return pick_lp_game_theory_move(game.board, int(game.player), legal)
        if kind == "minimax":
            return pick_minimax_alpha_beta_move(game.board, int(game.player), legal, depth=2, max_branch=10)
        return pick_heuristic_move(legal)

    @staticmethod
    def _one_hot_policy(action: int) -> np.ndarray:
        policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        policy[action] = 1.0
        return policy

    def begin_human_game_training(self):
        self._episode_steps.clear()
        self.training_status = None
        self._pending_guardrail_target = None
        self._pending_guardrail_weight = 0.0

    def record_human_game_step(self, game: Gomoku, action: int, policy: np.ndarray, player: int):
        self.record_online_step(game=game, action=action, player=player, policy=policy, is_human=False)

    def record_online_step(
        self,
        game: Gomoku,
        action: int,
        player: int,
        policy: np.ndarray | None = None,
        is_human: bool = False,
    ):
        state = game.canonical_board().astype(np.float32, copy=True)
        move_target = self._one_hot_policy(action) if policy is None else policy.astype(np.float32, copy=True)
        human_target = self._one_hot_policy(action) if is_human else np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)

        board_before = self._canonical_to_absolute_board(state, actor_player=player)
        offense_target, defense_target, attack_gain, defense_gain, opp_threat_before = self._build_step_targets(
            board_before=board_before,
            player=player,
            action=action,
        )
        reward = self._shape_step_reward(
            attack_gain=attack_gain,
            defense_gain=defense_gain,
            opp_threat_before=opp_threat_before,
            board_before=board_before,
            player=player,
            action=action,
        )
        if is_human:
            guard_target = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
            guard_weight = 0.0
        else:
            guard_target = (
                np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
                if self._pending_guardrail_target is None
                else self._pending_guardrail_target.astype(np.float32, copy=True)
            )
            guard_weight = float(self._pending_guardrail_weight)
            self._pending_guardrail_target = None
            self._pending_guardrail_weight = 0.0

        sample = {
            "state": state,
            "policy": move_target,
            "value": float(reward),
            "human": human_target,
            "offense": offense_target,
            "defense": defense_target,
            "guard": guard_target,
            "guard_weight": guard_weight,
            "is_human": is_human,
            "hardness": 1.0 + min(2.0, abs(float(reward)) * 2.0) + 0.8 * guard_weight,
        }
        self._episode_steps.append(EpisodeStep(sample=sample, actor_player=int(player)))
        for aug in self._augment_sample(sample):
            self.replay_buffer.append(aug)

        # Per-step RL update.
        self._train_online_from_replay(steps=max(1, self.online_updates_per_step))

    def finish_human_game_training(self, winner: int | None) -> str:
        if not self._episode_steps:
            self.training_status = "本局没有可训练样本。"
            return self.training_status
        if self.net is None:
            self.training_status = "当前智能体无模型，已跳过在线训练。"
            self._episode_steps.clear()
            return self.training_status
        winner = 0 if winner is None else int(winner)
        if self.online_optimizer is None:
            self.online_optimizer = optim.Adam(self.net.parameters(), lr=self.online_lr, weight_decay=1e-4)

        # Endgame value injection: reinforce final result on all recorded steps.
        terminal_added = 0
        for step in self._episode_steps:
            if int(winner) == 0:
                terminal = 0.0
            else:
                terminal = 1.0 if int(winner) == int(step.actor_player) else -1.0
            enriched = {**step.sample}
            enriched["value"] = float(np.clip(float(enriched["value"]) + 0.8 * float(terminal), -1.0, 1.0))
            enriched["hardness"] = float(enriched["hardness"]) + 0.8
            for aug in self._augment_sample(enriched):
                self.replay_buffer.append(aug)
            terminal_added += 1

        avg_loss = self._train_online_from_replay(steps=max(1, self.online_updates_endgame))
        self.net.eval()
        model_path = self._online_model_path()
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), model_path)
        self._episode_steps.clear()
        self.training_status = (
            f"逐步更新={self.online_updates_per_step}/step，终局更新={self.online_updates_endgame}，"
            f"终局注入样本={terminal_added}，回放池={len(self.replay_buffer)}，"
            f"损失={avg_loss:.4f}，已保存={model_path.name}"
        )
        return self.training_status

    def _online_model_path(self) -> Path:
        current = self.choices[self.choice_index].path
        if current is not None:
            return current
        return self.root / "models" / "gomoku_human_online.pt"

    @staticmethod
    def _normalized_target(actions: list[int]) -> np.ndarray:
        target = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        if not actions:
            return target
        prob = 1.0 / float(len(actions))
        for a in actions:
            target[int(a)] = prob
        return target

    @staticmethod
    def _canonical_to_absolute_board(state: np.ndarray, actor_player: int) -> np.ndarray:
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        board[state[0] > 0.5] = int(actor_player)
        board[state[1] > 0.5] = int(-actor_player)
        return board

    def _build_step_targets(
        self,
        board_before: np.ndarray,
        player: int,
        action: int,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        legal = np.argwhere(board_before == 0)
        offense_raw = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        defense_raw = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        if len(legal) == 0:
            return offense_raw, defense_raw, 0.0, 0.0, 0.0

        opp = -int(player)
        chosen_attack = 0.0
        chosen_defense = 0.0
        opp_base_cache: dict[tuple[int, int], float] = {}
        ar, ac = divmod(int(action), BOARD_SIZE)

        for rr, cc in legal:
            r = int(rr)
            c = int(cc)
            idx = r * BOARD_SIZE + c
            opp_base = self._local_motif_score(board_before, opp, r, c, DEFENSE_PENALTY_WEIGHTS)
            opp_base_cache[(r, c)] = opp_base

            board_before[r, c] = int(player)
            try:
                att_after = self._local_motif_score(board_before, int(player), r, c, ATTACK_WEIGHTS)
                opp_after = self._local_motif_score(board_before, opp, r, c, DEFENSE_PENALTY_WEIGHTS)
            finally:
                board_before[r, c] = 0

            attack_gain = max(0.0, att_after)
            defense_gain = max(0.0, opp_base - opp_after)
            offense_raw[idx] = float(attack_gain)
            defense_raw[idx] = float(defense_gain)
            if r == int(ar) and c == int(ac):
                chosen_attack = float(attack_gain)
                chosen_defense = float(defense_gain)

        offense_target = self._normalize_map(offense_raw)
        defense_target = self._normalize_map(defense_raw)
        opp_threat_before = float(max(opp_base_cache.values()) if opp_base_cache else 0.0)
        return offense_target, defense_target, float(chosen_attack), float(chosen_defense), opp_threat_before

    def _shape_step_reward(
        self,
        attack_gain: float,
        defense_gain: float,
        opp_threat_before: float,
        board_before: np.ndarray,
        player: int,
        action: int,
    ) -> float:
        # Reward if selected action creates threat/reduces threat, and encourage jump motifs contextually.
        reward = 0.0
        reward += 0.22 * np.tanh(attack_gain / 6.0)
        reward += 0.30 * np.tanh(defense_gain / 8.0)

        r, c = divmod(int(action), BOARD_SIZE)
        board_after = np.array(board_before, copy=True)
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board_after[r, c] == 0:
            board_after[r, c] = int(player)
        jump21, jump31 = self._jump_pattern_counts(board_after, int(player), int(r), int(c))
        connect_len = self._max_line_len(board_after, int(player), int(r), int(c))
        jump_bonus = 0.08 * float(jump21) + 0.14 * float(jump31)
        connect_bonus = 0.0
        if connect_len >= 4:
            connect_bonus += 0.10
        if connect_len >= 5:
            connect_bonus += 0.20

        # High urgency -> prefer direct connect/defense over jump motifs.
        if connect_len >= 4 or attack_gain > 6.0:
            jump_bonus *= 0.35
        if opp_threat_before > 6.0 or defense_gain > 0.0:
            jump_bonus *= 0.30

        reward += jump_bonus + connect_bonus

        if opp_threat_before > 6.0 and defense_gain <= 1e-6:
            reward -= 0.40
        elif defense_gain > 0.0:
            reward += 0.18

        if attack_gain <= 1e-6:
            reward -= 0.18
        else:
            reward += 0.12

        return float(np.clip(reward, -1.0, 1.0))

    @staticmethod
    def _local_motif_score(
        board: np.ndarray,
        player: int,
        r: int,
        c: int,
        weights: dict[str, float],
    ) -> float:
        feats = local_pattern_features(board, int(player), int(r), int(c))
        return float(sum(float(feats.get(k, 0)) * float(w) for k, w in weights.items()))

    @staticmethod
    def _normalize_map(raw: np.ndarray) -> np.ndarray:
        out = np.asarray(raw, dtype=np.float32).copy()
        s = float(out.sum())
        if s > 1e-8:
            out /= s
        return out

    @staticmethod
    def _jump_pattern_counts(board: np.ndarray, player: int, r: int, c: int) -> tuple[int, int]:
        jump21 = 0
        jump31 = 0
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            line: list[str] = []
            idx = 0
            rr, cc = int(r), int(c)
            while 0 <= rr - dr < BOARD_SIZE and 0 <= cc - dc < BOARD_SIZE:
                rr -= dr
                cc -= dc
            while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
                if rr == int(r) and cc == int(c):
                    idx = len(line)
                line.append(GomokuAgent._cell_symbol(board, rr, cc, int(player)))
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

    def _train_online_from_replay(self, steps: int) -> float:
        if self.net is None or self.online_optimizer is None or not self.replay_buffer:
            return 0.0

        was_training = bool(self.net.training)
        self.net.train()
        total_loss = 0.0
        steps = max(1, int(steps))
        batch_size = min(self.online_batch_size, len(self.replay_buffer))
        try:
            for _ in range(steps):
                batch = self._sample_replay_batch(batch_size)
                states = torch.tensor(np.stack([b["state"] for b in batch]), dtype=torch.float32, device=self.device)
                policy_target = torch.tensor(np.stack([b["policy"] for b in batch]), dtype=torch.float32, device=self.device)
                values = torch.tensor([b["value"] for b in batch], dtype=torch.float32, device=self.device)
                human_target = torch.tensor(np.stack([b["human"] for b in batch]), dtype=torch.float32, device=self.device)
                offense_target = torch.tensor(np.stack([b["offense"] for b in batch]), dtype=torch.float32, device=self.device)
                defense_target = torch.tensor(np.stack([b["defense"] for b in batch]), dtype=torch.float32, device=self.device)
                guard_target = torch.tensor(
                    np.stack(
                        [b.get("guard", np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)) for b in batch]
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                guard_weight = torch.tensor([float(b.get("guard_weight", 0.0)) for b in batch], dtype=torch.float32, device=self.device)

                self.online_optimizer.zero_grad()
                outputs = self.net(states)
                if isinstance(outputs, tuple) and len(outputs) == 6:
                    logits, value_pred, prior_logits, attack_logits, defense_logits, phase_logit = outputs
                elif isinstance(outputs, tuple) and len(outputs) == 3:
                    logits, value_pred, prior_logits = outputs
                    attack_logits = logits.detach()
                    defense_logits = logits.detach()
                    phase_logit = torch.zeros_like(value_pred)
                else:
                    logits, value_pred = outputs
                    prior_logits = logits.detach()
                    attack_logits = logits.detach()
                    defense_logits = logits.detach()
                    phase_logit = torch.zeros_like(value_pred)

                log_probs = F.log_softmax(logits, dim=1)
                probs = log_probs.exp()
                prior_log_probs = F.log_softmax(prior_logits, dim=1)
                attack_log_probs = F.log_softmax(attack_logits, dim=1)
                defense_log_probs = F.log_softmax(defense_logits, dim=1)

                policy_loss = -(policy_target * log_probs).sum(dim=1)
                value_loss = (value_pred - values) ** 2
                human_loss = -(human_target * log_probs).sum(dim=1)
                offense_loss = -(offense_target * attack_log_probs).sum(dim=1)
                defense_loss = -(defense_target * defense_log_probs).sum(dim=1)
                guard_loss = -(guard_target * log_probs).sum(dim=1)

                mixed_target = 0.55 * defense_target + 0.35 * offense_target + 0.10 * torch.clamp_min(policy_target, 0.0)
                mixed_sum = mixed_target.sum(dim=1, keepdim=True)
                fallback = torch.clamp_min(policy_target, 1e-8)
                fallback = fallback / fallback.sum(dim=1, keepdim=True)
                mixed_target = torch.where(mixed_sum > 0, mixed_target / torch.clamp_min(mixed_sum, 1e-8), fallback)
                prior_loss = -(mixed_target * prior_log_probs).sum(dim=1)
                prior_align = (probs * (log_probs - prior_log_probs)).sum(dim=1)

                has_human = (human_target.sum(dim=1) > 0).float()
                has_offense = (offense_target.sum(dim=1) > 0).float()
                has_defense = (defense_target.sum(dim=1) > 0).float()
                has_guard = (guard_target.sum(dim=1) > 0).float() * torch.clamp(guard_weight, min=0.0, max=1.0)

                phase_target = torch.full_like(values, 0.55)
                phase_target = torch.where((has_offense > 0) & (has_defense == 0), torch.full_like(values, 0.85), phase_target)
                phase_target = torch.where((has_defense > 0) & (has_offense == 0), torch.full_like(values, 0.15), phase_target)
                phase_target = torch.where((has_defense > 0) & (has_offense > 0), torch.full_like(values, 0.25), phase_target)
                phase_loss = F.binary_cross_entropy_with_logits(phase_logit, phase_target, reduction="none")

                total_per = (
                    policy_loss
                    + value_loss
                    + 0.20 * human_loss * has_human
                    + 0.55 * offense_loss * has_offense
                    + 0.70 * defense_loss * has_defense
                    + 0.85 * guard_loss * has_guard
                    + 0.35 * prior_loss
                    + 0.08 * prior_align
                    + 0.20 * phase_loss
                )
                # Hard samples are already up-weighted by prioritized sampling in _sample_replay_batch.
                # Avoid double-amplifying them again in loss scale.
                loss = total_per.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.online_optimizer.step()
                total_loss += float(loss.item())
            return total_loss / float(steps)
        finally:
            self.net.train(mode=was_training)

    def _sample_replay_batch(self, size: int) -> list[dict]:
        data = list(self.replay_buffer)
        weights = np.array([d["hardness"] for d in data], dtype=np.float64)
        total = float(weights.sum())
        if total <= 0:
            idx = np.random.choice(len(data), size=size, replace=len(data) < size)
        else:
            probs = weights / total
            idx = np.random.choice(len(data), size=size, replace=len(data) < size, p=probs)
        return [data[int(i)] for i in idx]

    def _augment_sample(self, sample: dict) -> list[dict]:
        out: list[dict] = []
        base_state = sample["state"]
        base_policy = sample["policy"].reshape(BOARD_SIZE, BOARD_SIZE)
        base_human = sample["human"].reshape(BOARD_SIZE, BOARD_SIZE)
        base_offense = sample["offense"].reshape(BOARD_SIZE, BOARD_SIZE)
        base_defense = sample["defense"].reshape(BOARD_SIZE, BOARD_SIZE)
        base_guard = sample.get("guard", np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)).reshape(BOARD_SIZE, BOARD_SIZE)

        for k in range(4):
            st_rot = np.rot90(base_state, k=k, axes=(1, 2)).copy()
            p_rot = np.rot90(base_policy, k=k).copy()
            h_rot = np.rot90(base_human, k=k).copy()
            o_rot = np.rot90(base_offense, k=k).copy()
            d_rot = np.rot90(base_defense, k=k).copy()
            g_rot = np.rot90(base_guard, k=k).copy()
            out.append(
                {
                    **sample,
                    "state": st_rot,
                    "policy": p_rot.reshape(-1),
                    "human": h_rot.reshape(-1),
                    "offense": o_rot.reshape(-1),
                    "defense": d_rot.reshape(-1),
                    "guard": g_rot.reshape(-1),
                }
            )

            st_flip = np.flip(st_rot, axis=2).copy()
            p_flip = np.fliplr(p_rot).copy()
            h_flip = np.fliplr(h_rot).copy()
            o_flip = np.fliplr(o_rot).copy()
            d_flip = np.fliplr(d_rot).copy()
            g_flip = np.fliplr(g_rot).copy()
            out.append(
                {
                    **sample,
                    "state": st_flip,
                    "policy": p_flip.reshape(-1),
                    "human": h_flip.reshape(-1),
                    "offense": o_flip.reshape(-1),
                    "defense": d_flip.reshape(-1),
                    "guard": g_flip.reshape(-1),
                }
            )
        return out

    def _rank_actions(self, game: Gomoku, actions: Iterable[int]) -> int:
        actions = list(dict.fromkeys(actions))
        if self.net is None:
            return self._fallback_center_pref(actions)
        with torch.no_grad():
            board_tensor = torch.from_numpy(game.canonical_board()).unsqueeze(0).to(self.device)
            outputs = self.net(board_tensor)
            if isinstance(outputs, tuple) and len(outputs) == 6:
                policy_logits, _, prior_logits, attack_logits, defense_logits, phase_logit = outputs
                phase = torch.sigmoid(phase_logit).unsqueeze(1)
                tactical_logits = phase * attack_logits + (1.0 - phase) * defense_logits
                policy_logits = policy_logits + 0.30 * prior_logits + 0.25 * tactical_logits
            elif isinstance(outputs, tuple) and len(outputs) == 3:
                policy_logits, _, prior_logits = outputs
                policy_logits = policy_logits + 0.30 * prior_logits
            else:
                policy_logits, _ = outputs
            logits = policy_logits.squeeze(0).detach().cpu().numpy()
        return max(actions, key=lambda a: float(logits[a]))

    @staticmethod
    def _fallback_center_pref(actions: Iterable[int]) -> int:
        return center_prefer_move(actions)

    def _immediate_winning_actions(self, board: np.ndarray, player: int) -> list[int]:
        actions: list[int] = []
        empties = np.argwhere(board == 0)
        for rr, cc in empties:
            r = int(rr)
            c = int(cc)
            if self._is_winning_move(board, player, r, c):
                actions.append(r * BOARD_SIZE + c)
        return actions

    def _open_three_block_actions(self, board: np.ndarray, player: int) -> list[int]:
        actions: set[int] = set()
        for coords in self._all_lines():
            symbols = "".join(self._cell_symbol(board, r, c, player) for r, c in coords)
            padded = "O" + symbols + "O"
            actions.update(self._open_three_actions_from_line(padded, coords))
        return sorted(actions)

    def _open_three_actions_from_line(self, padded: str, coords: list[tuple[int, int]]) -> set[int]:
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
                # Skip pseudo-live-three shape: O.XXX.O  (user notation: x.ooo.x)
                # In this shape, both extension points are flanked by opponent/boundary,
                # so we should not force guardrail intervention.
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
                        actions.add(r * BOARD_SIZE + c)
                start = i + 1
        return actions

    def _offense_four_actions(self, board: np.ndarray, player: int) -> list[int]:
        out: list[int] = []
        empties = np.argwhere(board == 0)
        for rr, cc in empties:
            r = int(rr)
            c = int(cc)
            if board[r, c] != 0:
                continue
            board[r, c] = int(player)
            try:
                if self._max_line_len(board, int(player), r, c) >= 4:
                    out.append(r * BOARD_SIZE + c)
            finally:
                board[r, c] = 0
        return out

    def _model_preferred_action(self, game: Gomoku) -> int | None:
        if self.net is None:
            return None
        legal = game.legal_moves()
        if not legal:
            return None
        with torch.no_grad():
            board_tensor = torch.from_numpy(game.canonical_board()).unsqueeze(0).to(self.device)
            outputs = self.net(board_tensor)
            if isinstance(outputs, tuple) and len(outputs) == 6:
                policy_logits, _, prior_logits, attack_logits, defense_logits, phase_logit = outputs
                phase = torch.sigmoid(phase_logit).unsqueeze(1)
                tactical_logits = phase * attack_logits + (1.0 - phase) * defense_logits
                fused = policy_logits + 0.30 * prior_logits + 0.25 * tactical_logits
            elif isinstance(outputs, tuple) and len(outputs) == 3:
                policy_logits, _, prior_logits = outputs
                fused = policy_logits + 0.30 * prior_logits
            else:
                fused, _ = outputs
            logits = fused.squeeze(0).detach().cpu().numpy()
        return int(max(legal, key=lambda a: float(logits[int(a)])))

    def _max_line_len(self, board: np.ndarray, player: int, r: int, c: int) -> int:
        best = 1
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            count = 1
            count += self._count_dir(board, int(player), r, c, dr, dc)
            count += self._count_dir(board, int(player), r, c, -dr, -dc)
            if count > best:
                best = count
        return best

    @staticmethod
    def _cell_symbol(board: np.ndarray, r: int, c: int, player: int) -> str:
        v = int(board[r, c])
        if v == int(player):
            return "X"
        if v == 0:
            return "."
        return "O"

    def _all_lines(self) -> list[list[tuple[int, int]]]:
        lines: list[list[tuple[int, int]]] = []
        n = BOARD_SIZE
        for r in range(n):
            lines.append([(r, c) for c in range(n)])
        for c in range(n):
            lines.append([(r, c) for r in range(n)])
        for c0 in range(n):
            lines.append(self._collect_line(0, c0, 1, 1))
        for r0 in range(1, n):
            lines.append(self._collect_line(r0, 0, 1, 1))
        for c0 in range(n):
            lines.append(self._collect_line(0, c0, 1, -1))
        for r0 in range(1, n):
            lines.append(self._collect_line(r0, n - 1, 1, -1))
        return [line for line in lines if len(line) >= 5]

    @staticmethod
    def _collect_line(r0: int, c0: int, dr: int, dc: int) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        r, c = r0, c0
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            out.append((r, c))
            r += dr
            c += dc
        return out

    @staticmethod
    def _is_winning_move(board: np.ndarray, player: int, r: int, c: int) -> bool:
        if board[r, c] != 0:
            return False
        board[r, c] = int(player)
        try:
            return GomokuAgent._has_five_from(board, int(player), r, c)
        finally:
            board[r, c] = 0

    @staticmethod
    def _has_five_from(board: np.ndarray, player: int, r: int, c: int) -> bool:
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            count = 1
            count += GomokuAgent._count_dir(board, int(player), r, c, dr, dc)
            count += GomokuAgent._count_dir(board, int(player), r, c, -dr, -dc)
            if count >= 5:
                return True
        return False

    @staticmethod
    def _count_dir(board: np.ndarray, player: int, r: int, c: int, dr: int, dc: int) -> int:
        nr, nc = r + dr, c + dc
        cnt = 0
        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and int(board[nr, nc]) == int(player):
            cnt += 1
            nr += dr
            nc += dc
        return cnt
