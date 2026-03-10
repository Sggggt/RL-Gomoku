"""
Entry script to run AlphaZero self-play training for 20x20 Gomoku.
Includes optional board playback visualization and anchor data to reduce regression risk.
"""

import os
from pathlib import Path

import numpy as np
import torch

from gomoku_logic import BOARD_SIZE, BLACK, WHITE, Gomoku
from alphazero import GomokuNet, MCTS, self_play_games, self_play_games_parallel, self_play_games_vs_heuristic, Trainer


TILE = 26
MARGIN = 28
GRID_SPAN = (BOARD_SIZE - 1) * TILE
WIDTH = MARGIN * 2 + GRID_SPAN + 240
HEIGHT = MARGIN * 2 + GRID_SPAN + 100

BG_COLOR = (245, 214, 133)
GRID_COLOR = (80, 60, 30)
BLACK_COLOR = (20, 20, 20)
WHITE_COLOR = (240, 240, 240)
TEXT_COLOR = (30, 30, 30)


def _count_dir(board: np.ndarray, player: int, r: int, c: int, dr: int, dc: int) -> int:
    n = board.shape[0]
    nr, nc = r + dr, c + dc
    cnt = 0
    while 0 <= nr < n and 0 <= nc < n and int(board[nr, nc]) == player:
        cnt += 1
        nr += dr
        nc += dc
    return cnt


def _has_five_from(board: np.ndarray, player: int, r: int, c: int) -> bool:
    n = board.shape[0]
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        cnt = 1
        cnt += _count_dir(board, player, r, c, dr, dc)
        cnt += _count_dir(board, player, r, c, -dr, -dc)
        if cnt >= 5:
            return True
    return False


def _immediate_winning_actions(board: np.ndarray, player: int) -> list[int]:
    acts: list[int] = []
    n = board.shape[0]
    empties = np.argwhere(board == 0)
    for r, c in empties:
        rr = int(r)
        cc = int(c)
        if board[rr, cc] != 0:
            continue
        board[rr, cc] = player
        try:
            if _has_five_from(board, player, rr, cc):
                acts.append(rr * n + cc)
        finally:
            board[rr, cc] = 0
    return acts


def _heuristic_action(game: Gomoku, player: int) -> int:
    legal = game.legal_moves()
    if not legal:
        raise ValueError("No legal move for heuristic.")
    win = _immediate_winning_actions(game.board, player)
    if win:
        return int(np.random.choice(win))
    block = _immediate_winning_actions(game.board, -player)
    if block:
        return int(np.random.choice(block))

    stones = np.argwhere(game.board != 0)
    center = (BOARD_SIZE - 1) / 2.0
    best_score = -1e9
    best_actions: list[int] = []
    for a in legal:
        r, c = divmod(int(a), BOARD_SIZE)
        center_score = -0.08 * (abs(r - center) + abs(c - center))
        if len(stones) == 0:
            prox_score = 0.0
        else:
            dmin = float(np.min(np.abs(stones[:, 0] - r) + np.abs(stones[:, 1] - c)))
            prox_score = -0.22 * dmin
        score = center_score + prox_score
        if score > best_score:
            best_score = score
            best_actions = [int(a)]
        elif score == best_score:
            best_actions.append(int(a))
    return int(np.random.choice(best_actions))


def _draw_demo_board(screen, font, game: Gomoku, mode: str, move_idx: int, status: str):
    import pygame

    screen.fill(BG_COLOR)

    for i in range(BOARD_SIZE):
        x = MARGIN + i * TILE
        y = MARGIN + i * TILE
        pygame.draw.line(screen, GRID_COLOR, (x, MARGIN), (x, MARGIN + GRID_SPAN), 1)
        pygame.draw.line(screen, GRID_COLOR, (MARGIN, y), (MARGIN + GRID_SPAN, y), 1)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = int(game.board[r, c])
            if v == 0:
                continue
            color = BLACK_COLOR if v == BLACK else WHITE_COLOR
            cx = MARGIN + c * TILE
            cy = MARGIN + r * TILE
            pygame.draw.circle(screen, color, (cx, cy), TILE // 2 - 2)
            pygame.draw.circle(screen, (50, 50, 50), (cx, cy), TILE // 2 - 2, 1)

    title = f"Playback | mode={mode} | move={move_idx}"
    screen.blit(font.render(title, True, TEXT_COLOR), (MARGIN, MARGIN + GRID_SPAN + 12))
    screen.blit(font.render(status, True, TEXT_COLOR), (MARGIN, MARGIN + GRID_SPAN + 40))


def visualize_demo_game(
    net,
    mode: str,
    device: str,
    posterior_stats: dict | None,
    posterior_scale: float = 0.25,
    simulations: int = 80,
    step_delay_ms: int = 120,
):
    try:
        import pygame
    except ImportError:
        print("Board visualization skipped: pygame is not installed.")
        return

    pygame.init()
    try:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Gomoku Self-Play Playback")
        font = pygame.font.SysFont("Consolas", 22)
        clock = pygame.time.Clock()

        game = Gomoku()
        mcts = MCTS(
            net,
            device=device,
            simulations=simulations,
            posterior_stats=posterior_stats,
            posterior_scale=posterior_scale,
        )
        move_idx = 0
        net_player = BLACK

        running = True
        while running and not game.finished:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if not running:
                break

            if mode == "heuristic" and game.player != net_player:
                action = _heuristic_action(game, game.player)
                status = "Heuristic move"
            else:
                policy = mcts.run(game, add_noise=False, temperature=0.0)
                action = int(np.argmax(policy))
                status = "Model/MCTS move"

            game.step(action)
            move_idx += 1
            _draw_demo_board(screen, font, game, mode, move_idx, status)
            pygame.display.flip()
            pygame.time.wait(step_delay_ms)
            clock.tick(60)

        if game.finished:
            if game.winner == BLACK:
                end_status = "Game finished: BLACK wins"
            elif game.winner == WHITE:
                end_status = "Game finished: WHITE wins"
            else:
                end_status = "Game finished: DRAW"
            _draw_demo_board(screen, font, game, mode, move_idx, end_status)
            pygame.display.flip()

            end_wait = 1800
            elapsed = 0
            while elapsed < end_wait:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        elapsed = end_wait
                        break
                pygame.time.wait(60)
                elapsed += 60
    finally:
        pygame.quit()


def run_selfplay(
    net,
    num_games: int,
    device: str,
    num_workers: int | None,
    mode: str = "selfplay",
    posterior_stats: dict | None = None,
    posterior_scale: float = 0.25,
):
    if mode == "heuristic":
        if (num_workers or 0) > 1:
            print("Heuristic mode currently runs single-process; ignoring parallel workers.")
        return self_play_games_vs_heuristic(
            net,
            num_games=num_games,
            device=device,
            simulations=150,
            posterior_stats=posterior_stats,
            posterior_scale=posterior_scale,
        )
    if num_workers is None or num_workers > 1:
        return self_play_games_parallel(
            net,
            num_games=num_games,
            num_workers=num_workers,
            device=device,
            simulations=150,
            batch_eval=(os.name != "nt"),
            posterior_stats=posterior_stats,
            posterior_scale=posterior_scale,
        )
    return self_play_games(
        net,
        num_games=num_games,
        device=device,
        simulations=150,
        posterior_stats=posterior_stats,
        posterior_scale=posterior_scale,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        num_games = int(input("Enter number of self-play games to generate: "))
    except ValueError:
        print("Invalid input. Using 10 games.")
        num_games = 10

    workers_input = input("Enter number of parallel workers (press Enter for auto): ").strip()
    num_workers = None
    if workers_input:
        try:
            num_workers = int(workers_input)
        except ValueError:
            print("Invalid workers input. Using auto selection.")
            num_workers = None

    mode_input = input("Select mode: [S] self-play, [H] heuristic-vs-model (default S): ").strip().lower()
    mode = "heuristic" if mode_input in {"h", "heuristic"} else "selfplay"
    viz_input = input("Enable board playback visualization before training? [y/N]: ").strip().lower()
    enable_viz = viz_input in {"y", "yes", "1", "true"}

    net = GomokuNet().to(device)
    model_path = Path("models/gomoku_alphazero.pt")
    posterior_stats = None
    has_baseline = model_path.exists()
    baseline_net = None

    if has_baseline:
        print(f"Loading existing model from {model_path}")
        state = torch.load(model_path, map_location=device)
        net.load_state_dict(state, strict=False)
        baseline_net = GomokuNet().to(device)
        baseline_net.load_state_dict(net.state_dict(), strict=False)
        baseline_net.eval()

    posterior_stats = Trainer.build_attention_posterior(
        net,
        base_std=0.01,
        prior_std=0.012,
        attention_std=0.02,
        phase_std=0.018,
    )
    print("Initialized fresh attention-aware posterior stats for this training run.")

    if enable_viz:
        print("Running one board playback demo before training...")
        visualize_demo_game(
            net,
            mode=mode,
            device=device,
            posterior_stats=posterior_stats,
            posterior_scale=0.25,
            simulations=80,
            step_delay_ms=120,
        )

    mode_text = "heuristic-vs-model games" if mode == "heuristic" else "self-play games"
    print(f"Generating {num_games} {mode_text} {'in parallel' if (num_workers or 0) != 1 and mode != 'heuristic' else ''}...")
    data = run_selfplay(
        net,
        num_games=num_games,
        device=device,
        num_workers=num_workers,
        mode=mode,
        posterior_stats=posterior_stats,
        posterior_scale=0.25,
    )
    print(f"Collected {len(data)} new training samples.")

    # Keep ~15% anchor samples from baseline strong model to reduce drift.
    anchor_data = []
    if baseline_net is not None and len(data) > 0:
        target_anchor = max(1, int(len(data) * 0.1765))
        anchor_games = max(1, int(num_games * 0.2))
        print(f"Generating anchor data from baseline model (target samples: {target_anchor})...")
        while len(anchor_data) < target_anchor:
            anchor_batch = run_selfplay(
                baseline_net,
                num_games=anchor_games,
                device=device,
                num_workers=1,
                mode=mode,
                posterior_stats=posterior_stats,
                posterior_scale=0.25,
            )
            if not anchor_batch:
                break
            anchor_data.extend(anchor_batch)
        anchor_data = anchor_data[:target_anchor]
        print(f"Collected {len(anchor_data)} anchor samples.")

    train_data = data + anchor_data
    print(f"Total training samples: {len(train_data)}")

    trainer = Trainer(
        lr=1e-3,
        batch_size=64,
        epochs=2,
        prior_weight=0.005,
        prior_align_weight=0.04,
        attack_head_weight=0.5,
        defense_head_weight=0.6,
        phase_weight=0.25,
        tactical_align_weight=0.08,
        hard_ratio=0.5,
    )
    trainer.set_posterior_stats(posterior_stats)
    trainer.align_posterior_to_net(net)
    avg_loss = trainer.train(net, train_data, device=device)
    print(f"Training finished. Avg loss: {avg_loss:.4f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
