# RL-Gomoku (AlphaZero 驱动的五子棋 AI)

基于 AlphaZero 算法实现的 20x20 五子棋强化学习项目，支持自我博弈训练、MCTS 搜索、GUI 人机对战，以及基于人类对局反馈的在线强化学习。

本项目**摒弃了纯离线打谱训练**，将专家战术知识（如活三、四三等绝杀阵型）无缝集成到了强化学习的 Reward 塑形（Reward Shaping）和启发式探索中，实现了模型在从零开始自我博弈时的高效收敛。

## 📁 核心模块说明

- **`alphazero/`**: 包含 AlphaZero 的核心算法实现，包括神经网络结构 (`GomokuNet`)、蒙特卡洛树搜索 (`MCTS`) 以及自我博弈/并行训练的管线编排。
- **`agent/`**: 包含各类对弈机器人的逻辑封装。既包括基于加载 `gomoku_alphazero.pt` 权重的 AI，也包括基于纯规则搜索的（无模型）启发式 AI；同时内置了在人机对局中记录样本、进行在线更新并保存权重的人类强化学习流程。
- **`gomoku_gui/`**: 基于 Pygame 实现的五子棋可视化界面。
- **`supervised/`**: **战术阵型规则库**。包含五子棋核心绝杀阵型（如活三、冲四等）的定义 (`pattern_templates.json`) 与评估逻辑 (`patterns.py`)。该模块直接为强化学习在自对弈时提供局面的局部特征 Reward 奖励，不进行独立的模型反向微调。

## 🚀 快速开始

### 1. 环境安装

本项目依赖少量的核心科学计算与深度学习库。请使用 `pip` 安装依赖：

```bash
pip install -r requirements.txt
```

（建议在安装了 CUDA 的机器上运行以获得最佳训练速度。PyTorch 会自动根据 `torch.cuda.is_available()` 决定是否启用 GPU）。

### 2. 强化学习自博弈训练

从零开始训练或继续微调现有模型，只需运行根目录的 `train.py`：

```bash
python train.py
```

**训练流程说明：**
1. 脚本会提示你输入计划生成的**自博弈对局数量**（例如输入 `20`）。
2. 输入期望的**并发 Worker 进程数**（支持多进程打谱以最大化压榨 CPU/GPU，直接回车将自动根据系统 CPU 分配）。
3. 选择对弈模式（标准 Self-play 或是对抗 Heuristic 启发式算法）。
4. （可选）是否在终端或弹窗预览一局可视化的 MCTS 打谱过程。
5. **对局生成与反向传播**：生成的对局样本将混入少量的 Anchor 历史数据送入大模型，进行 Policy 和 Value 头的训练。训练完成后的权重将自动保存至 `models/gomoku_alphazero.pt`。

**关于强化学习的优化点：**
* **奖励塑形 (Motif-based shaping)**：系统会在每步落子后调用 `supervised/patterns.py`，若模型下出了如“成四”等绝杀阵型会触发即时正向 Reward，漏防对方绝杀阵则接受强烈惩罚。
* **贝叶斯先验与注意力增强**：训练过程中策略网络不仅拟合 MCTS，还被正则化“靠近中心点”的先验常识分布（详细见 `Trainer` 参数中的 `prior_weight`）。

### 3. 人机对战 GUI 模式

当你训练好模型（`models/gomoku_alphazero.pt` 存在）后，或者仅仅想体验无模型启发式 AI 的对抗，可以启动 GUI：

```bash
python gomoku_gui/play_gui.py
```

- **交互操作**：鼠标左键点击棋盘落子。
- **切换模式**：点击 `切换模式` 按钮，可在以下模式间循环切换：
  1. 人人对战 (Human vs Human)
  2. 玩家执黑 vs AI 执白 (Human vs AI)
  3. 玩家执白 vs AI 执黑 (AI vs Human)
- **切换智能体**：点击 `切换模型` 按钮，可在以下策略间轮换：
  1. 神经网络 + MCTS 模型（`models/*.pt`）
  2. `Heuristic (No Model)`
  3. `GameTheory LP`
  4. `Minimax AlphaBeta`
- **在线训练开关**：点击 `训练开关` 按钮，可控制当前人机对局是否启用在线训练。
- **重置游戏**：点击 `重置棋局` 按钮即可清空棋盘重新开始。

### 4. Agent 支持的人类强化学习

`gomoku_gui/play_gui.py` 中的人机模式不只是“对战”，还支持 **Human-in-the-loop 在线强化学习**：

- 当启用在线训练且当前智能体为神经网络模型时，`GomokuAgent` 会在对局过程中同步记录 AI 落子与人类落子样本。
- 每一步样本都会结合战术阵型特征构造 shaped reward，包括进攻收益、防守收益、紧急拦截与 guardrail 目标。
- 对局结束后，agent 会把终局胜负结果回注到整局样本上，再从 replay buffer 中执行若干轮在线更新。
- 更新后的参数会自动保存回当前模型文件；如果当前不是神经网络模型，则不会执行在线训练。

这意味着你可以把真人对局直接作为训练信号，让 agent 在实战交互中持续微调，而不是只能依赖离线自博弈数据。

## 🧠 关于战术规则库 (supervised)

早期的版本包含独立的监督学习微调脚本。目前，独立的预训练监督数据生成流程（如原先的 `train_pattern_recognizer.py`）已被**移除**，以保持架构的极致精简。

现在的 `supervised/` 目录是一个纯粹的**棋理特征提取引擎**，它由 `samples/pattern_templates.json`（棋盘阵型描述）驱动，为强化学习及 Heuristic Agent 的落子提供先验评分与战术拦截指引，有效避免了 AlphaZero 早期“盲目试错”导致的漫长收敛期。
