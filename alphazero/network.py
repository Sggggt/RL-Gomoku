import torch
import torch.nn as nn
import torch.nn.functional as F

from gomoku_logic import BOARD_SIZE


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class GomokuNet(nn.Module):
    """
    Lightweight AlphaZero-style network.
    Input: (batch, 2, 20, 20) planes.
    Outputs:
      policy logits shape (batch, BOARD_SIZE * BOARD_SIZE)
      value in [-1, 1]
      prior logits shape (batch, BOARD_SIZE * BOARD_SIZE), learnable search prior
      attack attention logits shape (batch, BOARD_SIZE * BOARD_SIZE)
      defense attention logits shape (batch, BOARD_SIZE * BOARD_SIZE)
      phase logit shape (batch,), attack-vs-defense tendency
    """

    def __init__(self, channels: int = 64, num_blocks: int = 3):
        super().__init__()
        self.input_conv = nn.Conv2d(2, channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_blocks)])

        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        self.prior_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.prior_bn = nn.BatchNorm2d(2)
        self.prior_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        self.attack_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.attack_bn = nn.BatchNorm2d(2)
        self.attack_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        self.defense_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.defense_bn = nn.BatchNorm2d(2)
        self.defense_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)

        self.phase_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.phase_bn = nn.BatchNorm2d(1)
        self.phase_fc = nn.Linear(BOARD_SIZE * BOARD_SIZE, 1)

        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, channels)
        self.value_fc2 = nn.Linear(channels, 1)

    def forward(self, x):
        out = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.blocks:
            out = block(out)

        # policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # prior head
        pr = F.relu(self.prior_bn(self.prior_conv(out)))
        pr = pr.view(pr.size(0), -1)
        prior_logits = self.prior_fc(pr)

        # tactical attention heads
        a = F.relu(self.attack_bn(self.attack_conv(out)))
        a = a.view(a.size(0), -1)
        attack_logits = self.attack_fc(a)

        d = F.relu(self.defense_bn(self.defense_conv(out)))
        d = d.view(d.size(0), -1)
        defense_logits = self.defense_fc(d)

        ph = F.relu(self.phase_bn(self.phase_conv(out)))
        ph = ph.view(ph.size(0), -1)
        phase_logit = self.phase_fc(ph).squeeze(-1)

        # value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return policy_logits, value, prior_logits, attack_logits, defense_logits, phase_logit
