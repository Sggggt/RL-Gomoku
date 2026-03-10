import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from gomoku_logic import Gomoku, BOARD_SIZE


@dataclass
class Node:
    prior: float
    to_play: int  # player who will make a move at this node
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, "Node"] = field(default_factory=dict)
    is_expanded: bool = False

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(
        self,
        net=None,
        device="cpu",
        c_puct: float = 1.5,
        simulations: int = 200,
        evaluator=None,
        center_bias: float = 0.08,
        stone_proximity_bias: float = 0.55,
        distance_power: float = 1.25,
        dirichlet_alpha: float = 0.3,
        dirichlet_frac: float = 0.25,
        prior_mix: float = 0.35,
        tactical_mix: float = 0.3,
        posterior_stats: dict | None = None,
        posterior_scale: float = 0.5,
    ):
        """
        net: PyTorch model (may be None if using external evaluator)
        evaluator: callable(board_np) -> (policy_logits_np, value_float) for batched/shared inference
        """
        self.net = net
        self.device = device
        self.c_puct = c_puct
        self.simulations = simulations
        self.evaluator = evaluator
        self.center_bias = center_bias
        self.stone_proximity_bias = stone_proximity_bias
        self.distance_power = distance_power
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac
        self.prior_mix = prior_mix
        self.tactical_mix = tactical_mix
        self.posterior_stats = posterior_stats
        self.posterior_scale = posterior_scale

    def run(self, game: Gomoku, add_noise: bool = True, temperature: float = 1.0):
        original_state = self._apply_posterior_sample()
        root = Node(prior=1.0, to_play=game.player)
        try:
            self._expand(root, game)
            if add_noise:
                self._add_dirichlet_noise(root, game)

            for _ in range(self.simulations):
                self._simulate(game.clone(), root)

            visits = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
            for action, child in root.children.items():
                visits[action] = child.visit_count

            if temperature == 0:
                best = int(np.argmax(visits))
                policy = np.zeros_like(visits)
                policy[best] = 1.0
            else:
                inv_temp = 1.0 / max(float(temperature), 1e-3)
                v64 = visits.astype(np.float64, copy=False)
                positive = v64 > 0
                if not bool(np.any(positive)):
                    policy = np.ones_like(visits) / visits.size
                else:
                    # Stable form: policy ∝ exp(inv_temp * log(visits))
                    logits = np.full(v64.shape, -np.inf, dtype=np.float64)
                    logits[positive] = np.log(v64[positive]) * inv_temp
                    m = float(np.max(logits[positive]))
                    weights = np.zeros_like(v64)
                    weights[positive] = np.exp(logits[positive] - m)
                    z = float(weights.sum())
                    if not np.isfinite(z) or z <= 0.0:
                        policy = np.ones_like(visits) / visits.size
                    else:
                        policy = (weights / z).astype(np.float32, copy=False)
            return policy
        finally:
            if original_state is not None and self.net is not None:
                self.net.load_state_dict(original_state, strict=False)

    def _apply_posterior_sample(self):
        if self.net is None or not self.posterior_stats:
            return None
        mean = self.posterior_stats.get("mean", {})
        sq_mean = self.posterior_stats.get("sq_mean", {})
        count = int(self.posterior_stats.get("count", 0))
        if count <= 0 or not mean or not sq_mean:
            return None
        with torch.no_grad():
            original = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            sampled = {k: v.detach().clone() for k, v in original.items()}
            for name, mu in mean.items():
                if name not in sampled or name not in sq_mean:
                    continue
                tensor = sampled[name]
                if not torch.is_floating_point(tensor):
                    continue
                mu_t = mu.to(tensor.device, dtype=tensor.dtype)
                sq_t = sq_mean[name].to(tensor.device, dtype=tensor.dtype)
                var_t = torch.clamp(sq_t - mu_t * mu_t, min=1e-8)
                eps = torch.randn_like(mu_t)
                sampled[name] = mu_t + self.posterior_scale * torch.sqrt(var_t) * eps
            self.net.load_state_dict(sampled, strict=False)
            return original

    def _simulate(self, game: Gomoku, node: Node):
        path = [node]
        # Selection
        while node.is_expanded and node.children:
            action, node = self._select_child(node)
            game.step(action)
            path.append(node)

        # Evaluation / expansion
        value = self._evaluate(game, node)
        # Backpropagate
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value  # flip perspective

    def _select_child(self, node: Node):
        total_visits = math.sqrt(sum(child.visit_count for child in node.children.values()) + 1e-8)
        best_score = -1e9
        best_action = None
        best_child = None
        for action, child in node.children.items():
            prior = child.prior
            q = child.value()
            u = self.c_puct * prior * total_visits / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def _expand(self, node: Node, game: Gomoku):
        node.is_expanded = True
        if game.finished:
            return
        legal = game.legal_moves()
        if self.evaluator is None:
            with torch.no_grad():
                board_tensor = torch.from_numpy(game.canonical_board()).unsqueeze(0).to(self.device)
                outputs = self.net(board_tensor)
                if isinstance(outputs, tuple) and len(outputs) == 6:
                    policy_logits, value, prior_logits, attack_logits, defense_logits, phase_logit = outputs
                    prior_logits = prior_logits.squeeze(0)
                    attack_logits = attack_logits.squeeze(0)
                    defense_logits = defense_logits.squeeze(0)
                    phase = float(torch.sigmoid(phase_logit.squeeze(0)).item())
                elif isinstance(outputs, tuple) and len(outputs) == 3:
                    policy_logits, value, prior_logits = outputs
                    prior_logits = prior_logits.squeeze(0)
                    attack_logits = None
                    defense_logits = None
                    phase = 0.5
                else:
                    policy_logits, value = outputs
                    prior_logits = None
                    attack_logits = None
                    defense_logits = None
                    phase = 0.5
                policy_logits = policy_logits.squeeze(0)
                value = value.item()
            if prior_logits is not None:
                fused_logits = policy_logits + self.prior_mix * prior_logits
            else:
                fused_logits = policy_logits
            if attack_logits is not None and defense_logits is not None:
                tactical_logits = phase * attack_logits + (1.0 - phase) * defense_logits
                fused_logits = fused_logits + self.tactical_mix * tactical_logits
            policy = F.softmax(fused_logits, dim=0).cpu().numpy()
        else:
            eval_out = self.evaluator(game.canonical_board())
            if isinstance(eval_out, tuple) and len(eval_out) == 6:
                policy_logits_np, value, prior_logits_np, attack_logits_np, defense_logits_np, phase_val = eval_out
                phase = float(phase_val)
                fused = (
                    torch.from_numpy(policy_logits_np)
                    + self.prior_mix * torch.from_numpy(prior_logits_np)
                    + self.tactical_mix
                    * (
                        phase * torch.from_numpy(attack_logits_np)
                        + (1.0 - phase) * torch.from_numpy(defense_logits_np)
                    )
                )
            elif isinstance(eval_out, tuple) and len(eval_out) == 3:
                policy_logits_np, value, prior_logits_np = eval_out
                fused = torch.from_numpy(policy_logits_np) + self.prior_mix * torch.from_numpy(prior_logits_np)
            else:
                policy_logits_np, value = eval_out
                fused = torch.from_numpy(policy_logits_np)
            policy = F.softmax(fused, dim=0).numpy()
        policy_mask = np.zeros_like(policy)
        policy_mask[legal] = policy[legal]
        if policy_mask.sum() == 0:
            policy_mask[legal] = 1.0 / len(legal)
        # Apply spatial priors: center preference + proximity to existing stones penalty
        spatial = self._spatial_bias(game, legal)
        policy_mask[legal] = policy_mask[legal] * spatial
        if policy_mask.sum() == 0:
            policy_mask[legal] = 1.0 / len(legal)
        else:
            policy_mask = policy_mask / policy_mask.sum()
        for action in legal:
            node.children[action] = Node(prior=float(policy_mask[action]), to_play=-game.player)
        return value

    def _evaluate(self, game: Gomoku, node: Node) -> float:
        if game.finished:
            if game.winner == 0:
                return 0.0
            return 1.0 if game.winner == node.to_play else -1.0
        value = self._expand(node, game)
        return value

    def _add_dirichlet_noise(self, root: Node, game: Gomoku):
        actions = list(root.children.keys())
        if not actions:
            return
        local_actions = self._local_actions(game, actions, radius=1)
        if not local_actions:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(local_actions))
        for a, n in zip(local_actions, noise):
            root.children[a].prior = root.children[a].prior * (1 - self.dirichlet_frac) + n * self.dirichlet_frac

    @staticmethod
    def _move_count(game: Gomoku) -> int:
        return int((game.board != 0).sum())

    def _local_actions(self, game: Gomoku, actions: list[int], radius: int = 1) -> list[int]:
        """Return actions within `radius` of last move; if no last move, center 3x3."""
        if game.last_move is None:
            center = BOARD_SIZE // 2
            allowed = set(
                r * BOARD_SIZE + c
                for r in range(center - radius, center + radius + 1)
                for c in range(center - radius, center + radius + 1)
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE
            )
        else:
            lr, lc = game.last_move
            allowed = set(
                r * BOARD_SIZE + c
                for r in range(lr - radius, lr + radius + 1)
                for c in range(lc - radius, lc + radius + 1)
                if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE
            )
        return [a for a in actions if a in allowed]

    def _spatial_bias(self, game: Gomoku, legal: list[int]) -> np.ndarray:
        """
        Bias factor favoring center and proximity to existing stones.
        Returns array aligned with legal actions.
        """
        center = (BOARD_SIZE - 1) / 2
        board = game.board
        stones = np.argwhere(board != 0)
        spatial = np.ones(len(legal), dtype=np.float32)
        for i, action in enumerate(legal):
            r, c = divmod(action, BOARD_SIZE)
            dist_center = abs(r - center) + abs(c - center)
            bias_c = math.exp(-self.center_bias * dist_center)
            if len(stones) == 0:
                bias_s = 1.0
            else:
                # Manhattan distance to nearest stone
                dmin = np.min(np.abs(stones[:, 0] - r) + np.abs(stones[:, 1] - c))
                bias_s = math.exp(-self.stone_proximity_bias * (float(dmin) ** self.distance_power))
                if dmin >= 4:
                    bias_s *= 0.35
            spatial[i] = bias_c * bias_s
        # avoid zeros
        spatial = spatial + 1e-8
        return spatial
