from typing import Any, List
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from gomoku_logic import BOARD_SIZE


class Trainer:
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 1,
        prior_weight: float = 0.005,
        prior_sigma: float = 6.0,
        replay_capacity: int = 50000,
        hard_sample_weight: float = 2.5,
        hard_ratio: float = 0.5,
        offense_weight: float = 0.2,
        defense_weight: float = 0.3,
        prior_head_weight: float = 0.35,
        prior_align_weight: float = 0.04,
        attack_head_weight: float = 0.45,
        defense_head_weight: float = 0.5,
        phase_weight: float = 0.2,
        tactical_align_weight: float = 0.05,
        guard_weight: float = 0.85,
        augment: bool = True,
        posterior_collect: bool = True,
        posterior_start_frac: float = 0.7,
        posterior_interval: int = 20,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.prior_weight = prior_weight
        self.prior_sigma = prior_sigma
        self.replay_capacity = replay_capacity
        self.hard_sample_weight = hard_sample_weight
        self.hard_ratio = max(0.0, min(1.0, hard_ratio))
        self.offense_weight = offense_weight
        self.defense_weight = defense_weight
        self.prior_head_weight = prior_head_weight
        self.prior_align_weight = prior_align_weight
        self.attack_head_weight = attack_head_weight
        self.defense_head_weight = defense_head_weight
        self.phase_weight = phase_weight
        self.tactical_align_weight = tactical_align_weight
        self.guard_weight = guard_weight
        self.augment = augment
        self.posterior_collect = posterior_collect
        self.posterior_start_frac = posterior_start_frac
        self.posterior_interval = max(1, posterior_interval)

        self.replay_buffer: deque[dict] = deque(maxlen=replay_capacity)
        self.center_prior_flat = self._make_center_prior()
        self.posterior_mean: dict[str, torch.Tensor] = {}
        self.posterior_sq_mean: dict[str, torch.Tensor] = {}
        self.posterior_count: int = 0
        self.last_epoch_losses: list[float] = []

    def train(self, net: nn.Module, data: List[Any], device: str = "cpu"):
        if not data:
            self.last_epoch_losses = []
            return 0.0

        net.train()
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        normalized = [self._normalize_sample(d) for d in data]
        for sample in normalized:
            for aug in self._augment_sample(sample):
                self.replay_buffer.append(aug)
        if not self.replay_buffer:
            self.last_epoch_losses = []
            return 0.0

        steps_in_epoch = max(1, len(self.replay_buffer) // self.batch_size)
        total_planned_steps = max(1, self.epochs * steps_in_epoch)
        freeze_steps = int(total_planned_steps * 0.2)
        posterior_start_step = int(total_planned_steps * self.posterior_start_frac)
        global_step = 0

        effective_hard_weight = self.hard_sample_weight
        prev_normal_policy = None
        total_loss = 0.0
        total_steps = 0
        self.last_epoch_losses = []

        for _ in trange(self.epochs, desc="Training epochs", ncols=80):
            epoch_normal_policy_vals: list[float] = []
            epoch_loss_sum = 0.0
            epoch_steps = 0
            for _ in range(steps_in_epoch):
                self._set_prior_head_trainable(net, trainable=(global_step >= freeze_steps))

                batch = self._sample_replay_batch(min(self.batch_size, len(self.replay_buffer)))
                batch_states = torch.tensor(np.stack([b["state"] for b in batch]), dtype=torch.float32, device=device)
                batch_policy_target = torch.tensor(np.stack([b["policy"] for b in batch]), dtype=torch.float32, device=device)
                batch_value_target = torch.tensor([b["value"] for b in batch], dtype=torch.float32, device=device)
                batch_offense_target = torch.tensor(np.stack([b["offense"] for b in batch]), dtype=torch.float32, device=device)
                batch_defense_target = torch.tensor(np.stack([b["defense"] for b in batch]), dtype=torch.float32, device=device)
                batch_prior_target = torch.tensor(np.stack([b["prior_target"] for b in batch]), dtype=torch.float32, device=device)
                batch_phase_target = torch.tensor([b["phase_target"] for b in batch], dtype=torch.float32, device=device)
                batch_guard_target = torch.tensor(np.stack([b["guard"] for b in batch]), dtype=torch.float32, device=device)
                batch_guard_weight = torch.tensor([float(b.get("guard_weight", 0.0)) for b in batch], dtype=torch.float32, device=device)
                hard_mask = torch.tensor([1.0 if b["is_hard"] else 0.0 for b in batch], dtype=torch.float32, device=device)
                batch_weights = torch.where(hard_mask > 0, torch.full_like(hard_mask, effective_hard_weight), torch.ones_like(hard_mask))

                optimizer.zero_grad()
                outputs = net(batch_states)
                if isinstance(outputs, tuple) and len(outputs) == 6:
                    policy_logits, value_pred, prior_logits, attack_logits, defense_logits, phase_logit = outputs
                elif isinstance(outputs, tuple) and len(outputs) == 3:
                    policy_logits, value_pred, prior_logits = outputs
                    attack_logits = policy_logits.detach()
                    defense_logits = policy_logits.detach()
                    phase_logit = torch.zeros_like(value_pred)
                else:
                    policy_logits, value_pred = outputs
                    prior_logits = policy_logits.detach()
                    attack_logits = policy_logits.detach()
                    defense_logits = policy_logits.detach()
                    phase_logit = torch.zeros_like(value_pred)

                policy_log_probs = F.log_softmax(policy_logits, dim=1)
                policy_probs = policy_log_probs.exp()
                prior_log_probs = F.log_softmax(prior_logits, dim=1)
                attack_log_probs = F.log_softmax(attack_logits, dim=1)
                defense_log_probs = F.log_softmax(defense_logits, dim=1)

                policy_loss_per = -(batch_policy_target * policy_log_probs).sum(dim=1)
                value_loss_per = (value_pred - batch_value_target) ** 2
                offense_loss_per = -(batch_offense_target * policy_log_probs).sum(dim=1)
                defense_loss_per = -(batch_defense_target * policy_log_probs).sum(dim=1)
                guard_loss_per = -(batch_guard_target * policy_log_probs).sum(dim=1)
                attack_head_loss_per = -(batch_offense_target * attack_log_probs).sum(dim=1)
                defense_head_loss_per = -(batch_defense_target * defense_log_probs).sum(dim=1)
                phase_loss_per = F.binary_cross_entropy_with_logits(phase_logit, batch_phase_target, reduction="none")

                has_offense = (batch_offense_target.sum(dim=1) > 0).float()
                has_defense = (batch_defense_target.sum(dim=1) > 0).float()
                has_prior = (batch_prior_target.sum(dim=1) > 0).float()
                has_guard = (batch_guard_target.sum(dim=1) > 0).float() * torch.clamp(batch_guard_weight, min=0.0, max=1.0)

                prior = self._batch_prior(batch_states, device)
                kl_prior_per = (policy_probs * (policy_log_probs - torch.log(prior + 1e-8))).sum(dim=1)
                prior_head_loss_per = -(batch_prior_target * prior_log_probs).sum(dim=1)
                align_loss_per = (policy_probs * (policy_log_probs - prior_log_probs)).sum(dim=1)
                phase = torch.sigmoid(phase_logit).unsqueeze(1)
                tactical_logits = phase * attack_logits + (1.0 - phase) * defense_logits
                tactical_log_probs = F.log_softmax(tactical_logits, dim=1)
                tactical_align_loss_per = (policy_probs * (policy_log_probs - tactical_log_probs)).sum(dim=1)

                loss_per = (
                    policy_loss_per
                    + value_loss_per
                    + self.offense_weight * offense_loss_per * has_offense
                    + self.defense_weight * defense_loss_per * has_defense
                    + self.prior_head_weight * prior_head_loss_per * has_prior
                    + self.prior_align_weight * align_loss_per
                    + self.attack_head_weight * attack_head_loss_per * has_offense
                    + self.defense_head_weight * defense_head_loss_per * has_defense
                    + self.guard_weight * guard_loss_per * has_guard
                    + self.phase_weight * phase_loss_per
                    + self.tactical_align_weight * tactical_align_loss_per
                    + self.prior_weight * kl_prior_per
                )
                loss = (loss_per * batch_weights).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()

                normal_mask = (hard_mask == 0)
                if bool(normal_mask.any()):
                    epoch_normal_policy_vals.append(float(policy_loss_per[normal_mask].mean().item()))

                total_loss += float(loss.item())
                total_steps += 1
                epoch_loss_sum += float(loss.item())
                epoch_steps += 1

                if (
                    self.posterior_collect
                    and global_step >= posterior_start_step
                    and (global_step - posterior_start_step) % self.posterior_interval == 0
                ):
                    self._update_posterior_stats(net)
                global_step += 1

            if epoch_normal_policy_vals:
                cur_normal_policy = float(np.mean(epoch_normal_policy_vals))
                if prev_normal_policy is not None and cur_normal_policy > prev_normal_policy * 1.05:
                    effective_hard_weight = max(1.0, effective_hard_weight * 0.8)
                prev_normal_policy = cur_normal_policy
            self.last_epoch_losses.append(epoch_loss_sum / max(1, epoch_steps))

        self._set_prior_head_trainable(net, trainable=True)
        return total_loss / max(1, total_steps)

    def _update_posterior_stats(self, net: nn.Module):
        with torch.no_grad():
            new_count = self.posterior_count + 1
            for name, param in net.named_parameters():
                if not torch.is_floating_point(param):
                    continue
                p = param.detach().cpu().float()
                if self.posterior_count == 0 or name not in self.posterior_mean or name not in self.posterior_sq_mean:
                    self.posterior_mean[name] = p.clone()
                    self.posterior_sq_mean[name] = p.pow(2)
                else:
                    mean = self.posterior_mean[name]
                    sq = self.posterior_sq_mean[name]
                    mean += (p - mean) / float(new_count)
                    sq += (p.pow(2) - sq) / float(new_count)
                    self.posterior_mean[name] = mean
                    self.posterior_sq_mean[name] = sq
            self.posterior_count = new_count

    def get_posterior_stats(self) -> dict[str, Any]:
        return {
            "count": int(self.posterior_count),
            "mean": {k: v.clone() for k, v in self.posterior_mean.items()},
            "sq_mean": {k: v.clone() for k, v in self.posterior_sq_mean.items()},
        }

    @staticmethod
    def build_attention_posterior(
        net: nn.Module,
        base_std: float = 0.01,
        prior_std: float = 0.012,
        attention_std: float = 0.02,
        phase_std: float = 0.018,
    ) -> dict[str, Any]:
        """
        Build a fresh Gaussian posterior around current parameters.
        Attention-related heads use dedicated std to better fit upgraded architecture.
        """
        mean: dict[str, torch.Tensor] = {}
        sq_mean: dict[str, torch.Tensor] = {}
        for name, param in net.named_parameters():
            if not torch.is_floating_point(param):
                continue
            mu = param.detach().cpu().float().clone()
            if name.startswith("attack_") or name.startswith("defense_"):
                std = attention_std
            elif name.startswith("phase_"):
                std = phase_std
            elif name.startswith("prior_"):
                std = prior_std
            else:
                std = base_std
            mean[name] = mu
            sq_mean[name] = mu.pow(2) + (float(std) ** 2)
        return {"count": 1, "mean": mean, "sq_mean": sq_mean}

    def set_posterior_stats(self, stats: dict[str, Any] | None):
        if not stats:
            return
        count = int(stats.get("count", 0))
        mean = stats.get("mean", {})
        sq_mean = stats.get("sq_mean", {})
        if count <= 0 or not isinstance(mean, dict) or not isinstance(sq_mean, dict):
            return
        self.posterior_count = count
        self.posterior_mean = {str(k): v.detach().cpu().float().clone() for k, v in mean.items()}
        self.posterior_sq_mean = {str(k): v.detach().cpu().float().clone() for k, v in sq_mean.items()}

    def align_posterior_to_net(self, net: nn.Module):
        valid_names = {name for name, p in net.named_parameters() if torch.is_floating_point(p)}
        self.posterior_mean = {k: v for k, v in self.posterior_mean.items() if k in valid_names}
        self.posterior_sq_mean = {k: v for k, v in self.posterior_sq_mean.items() if k in valid_names}

    def _set_prior_head_trainable(self, net: nn.Module, trainable: bool):
        for name, param in net.named_parameters():
            if name.startswith("prior_"):
                param.requires_grad = trainable

    def _normalize_sample(self, sample: Any) -> dict:
        if isinstance(sample, dict):
            state = np.asarray(sample["state"], dtype=np.float32)
            policy = np.asarray(sample["policy"], dtype=np.float32)
            value = float(sample["value"])
            offense = np.asarray(sample.get("offense", np.zeros_like(policy)), dtype=np.float32)
            defense = np.asarray(sample.get("defense", np.zeros_like(policy)), dtype=np.float32)
            prior_target = np.asarray(sample.get("prior_target", policy), dtype=np.float32)
            phase_target = float(sample.get("phase_target", 0.5))
            guard = np.asarray(sample.get("guard", np.zeros_like(policy)), dtype=np.float32)
            guard_w = float(sample.get("guard_weight", 0.0))
            is_hard = float(sample.get("hardness", 1.0)) > 1.0
            return {
                "state": state,
                "policy": policy,
                "value": value,
                "offense": offense,
                "defense": defense,
                "prior_target": prior_target,
                "phase_target": phase_target,
                "guard": guard,
                "guard_weight": guard_w,
                "is_hard": bool(is_hard),
            }

        state, policy, value = sample
        policy_arr = np.asarray(policy, dtype=np.float32)
        zeros = np.zeros_like(policy_arr, dtype=np.float32)
        return {
            "state": np.asarray(state, dtype=np.float32),
            "policy": policy_arr,
            "value": float(value),
            "offense": zeros.copy(),
            "defense": zeros.copy(),
            "prior_target": policy_arr.copy(),
            "phase_target": 0.5,
            "guard": zeros.copy(),
            "guard_weight": 0.0,
            "is_hard": False,
        }

    def _sample_replay_batch(self, size: int) -> list[dict]:
        data = list(self.replay_buffer)
        hard = [d for d in data if d["is_hard"]]
        normal = [d for d in data if not d["is_hard"]]

        if hard and normal:
            hard_n = int(round(size * self.hard_ratio))
            hard_n = min(size, max(1, hard_n))
            normal_n = size - hard_n
            if normal_n <= 0:
                normal_n = 1
                hard_n = size - 1
            hard_idx = np.random.choice(len(hard), size=hard_n, replace=len(hard) < hard_n)
            normal_idx = np.random.choice(len(normal), size=normal_n, replace=len(normal) < normal_n)
            batch = [hard[int(i)] for i in hard_idx] + [normal[int(i)] for i in normal_idx]
            np.random.shuffle(batch)
            return batch

        src = hard if hard else normal
        idx = np.random.choice(len(src), size=size, replace=len(src) < size)
        return [src[int(i)] for i in idx]

    def _augment_sample(self, sample: dict) -> list[dict]:
        if not self.augment:
            return [sample]

        out: list[dict] = []
        base_state = sample["state"]
        base_policy = sample["policy"].reshape(BOARD_SIZE, BOARD_SIZE)
        base_offense = sample["offense"].reshape(BOARD_SIZE, BOARD_SIZE)
        base_defense = sample["defense"].reshape(BOARD_SIZE, BOARD_SIZE)
        base_prior = sample["prior_target"].reshape(BOARD_SIZE, BOARD_SIZE)
        base_guard = sample["guard"].reshape(BOARD_SIZE, BOARD_SIZE)

        for k in range(4):
            st_rot = np.rot90(base_state, k=k, axes=(1, 2)).copy()
            p_rot = np.rot90(base_policy, k=k).copy()
            o_rot = np.rot90(base_offense, k=k).copy()
            d_rot = np.rot90(base_defense, k=k).copy()
            pr_rot = np.rot90(base_prior, k=k).copy()
            g_rot = np.rot90(base_guard, k=k).copy()
            out.append(
                {
                    **sample,
                    "state": st_rot,
                    "policy": p_rot.reshape(-1),
                    "offense": o_rot.reshape(-1),
                    "defense": d_rot.reshape(-1),
                    "prior_target": pr_rot.reshape(-1),
                    "guard": g_rot.reshape(-1),
                }
            )

            st_flip = np.flip(st_rot, axis=2).copy()
            p_flip = np.fliplr(p_rot).copy()
            o_flip = np.fliplr(o_rot).copy()
            d_flip = np.fliplr(d_rot).copy()
            pr_flip = np.fliplr(pr_rot).copy()
            g_flip = np.fliplr(g_rot).copy()
            out.append(
                {
                    **sample,
                    "state": st_flip,
                    "policy": p_flip.reshape(-1),
                    "offense": o_flip.reshape(-1),
                    "defense": d_flip.reshape(-1),
                    "prior_target": pr_flip.reshape(-1),
                    "guard": g_flip.reshape(-1),
                }
            )

        return out

    def _make_center_prior(self):
        xs, ys = torch.meshgrid(
            torch.arange(BOARD_SIZE, dtype=torch.float32),
            torch.arange(BOARD_SIZE, dtype=torch.float32),
            indexing="ij",
        )
        cx = cy = (BOARD_SIZE - 1) / 2
        dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
        prior = torch.exp(-dist2 / (2 * self.prior_sigma**2))
        prior = prior / prior.sum()
        return prior.view(1, -1)

    def _batch_prior(self, batch_states: torch.Tensor, device: str):
        bsz = batch_states.size(0)
        dev = torch.device(device)
        if self.center_prior_flat.device != dev:
            center = self.center_prior_flat.to(dev)
        else:
            center = self.center_prior_flat
        occupancy = batch_states.sum(dim=1)
        empties = (occupancy == 0).view(bsz, -1).float()
        masked = center * empties + 1e-8
        masked = masked / masked.sum(dim=1, keepdim=True)
        return masked
