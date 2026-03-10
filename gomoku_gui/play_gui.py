"""
Gomoku GUI simulator (20x20) with modes:
0) human vs human
1) human (black) vs AI (white)
2) AI (black) vs human (white)

Human-vs-AI games support online updates: after each finished game,
AI samples from that game are used to update model parameters and save the model.
"""

from pathlib import Path
import sys

import pygame
import torch

# Ensure project root is on path for shared modules.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gomoku_logic import Gomoku, BOARD_SIZE, BLACK, WHITE
from agent import GomokuAgent

TILE = 30
MARGIN = 40
BOARD_PIXELS = BOARD_SIZE * TILE
GRID_SPAN = (BOARD_SIZE - 1) * TILE
SIDE_PANEL = 260
WIDTH = BOARD_PIXELS + 2 * MARGIN + SIDE_PANEL
HEIGHT = BOARD_PIXELS + 2 * MARGIN + 140

BG_COLOR = (255, 235, 59)
GRID_COLOR = (220, 30, 30)
BLACK_COLOR = (20, 20, 20)
WHITE_COLOR = (240, 240, 240)
TEXT_COLOR = (0, 0, 0)


class Button:
    def __init__(self, rect: pygame.Rect, label: str, action):
        self.rect = rect
        self.label = label
        self.action = action

    def draw(self, screen, font):
        pygame.draw.rect(screen, (240, 240, 240), self.rect, border_radius=6)
        pygame.draw.rect(screen, (120, 120, 120), self.rect, width=2, border_radius=6)
        text = font.render(self.label, True, TEXT_COLOR)
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)

    def handle(self, pos):
        return self.rect.collidepoint(pos)


def load_font(size: int):
    candidates = [
        "SimHei",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "PingFang SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    for name in candidates:
        try:
            font = pygame.font.SysFont(name, size)
            if font.size("A")[0] > 0:
                return font
        except Exception:
            continue
    return pygame.font.Font(None, size)


def draw_board(
    screen,
    font,
    train_font,
    game: Gomoku,
    winner_text: str | None,
    mode_text: str,
    train_text: str,
):
    screen.fill(BG_COLOR)
    for i in range(BOARD_SIZE):
        start = (MARGIN + i * TILE, MARGIN)
        end = (MARGIN + i * TILE, MARGIN + GRID_SPAN)
        pygame.draw.line(screen, GRID_COLOR, start, end, 2)
        start = (MARGIN, MARGIN + i * TILE)
        end = (MARGIN + GRID_SPAN, MARGIN + i * TILE)
        pygame.draw.line(screen, GRID_COLOR, start, end, 2)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            val = game.board[r, c]
            if val == 0:
                continue
            color = BLACK_COLOR if val == BLACK else WHITE_COLOR
            center = (MARGIN + c * TILE, MARGIN + r * TILE)
            pygame.draw.circle(screen, color, center, TILE // 2 - 2)
            pygame.draw.circle(screen, (50, 50, 50), center, TILE // 2 - 2, 1)

    status = "黑棋落子" if game.player == BLACK else "白棋落子"
    if winner_text:
        status = winner_text + " - 点击重置"
    text_surf = font.render(status, True, TEXT_COLOR)
    screen.blit(text_surf, (MARGIN, MARGIN + BOARD_PIXELS + 10))

    mode_surf = font.render(mode_text, True, TEXT_COLOR)
    screen.blit(mode_surf, (MARGIN, MARGIN + BOARD_PIXELS + 35))

    if train_text:
        train_surf = train_font.render(train_text, True, TEXT_COLOR)
        screen.blit(train_surf, (MARGIN, MARGIN + BOARD_PIXELS + 60))


def click_to_action(pos) -> int | None:
    x, y = pos
    left = MARGIN
    top = MARGIN
    right = MARGIN + GRID_SPAN
    bottom = MARGIN + GRID_SPAN
    if x < left or x > right or y < top or y > bottom:
        return None
    c = int((x - MARGIN) / TILE + 0.5)
    r = int((y - MARGIN) / TILE + 0.5)
    if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
        return None
    return r * BOARD_SIZE + c


def handle_click(game: Gomoku, pos) -> str | None:
    action = click_to_action(pos)
    if action is None:
        return None
    try:
        result = game.step(action)
    except ValueError:
        return None
    return game_result_text(result.winner, result.finished)


def game_result_text(winner: int | None, finished: bool) -> str | None:
    if not finished:
        return None
    if winner == BLACK:
        return "黑棋胜"
    if winner == WHITE:
        return "白棋胜"
    return "平局"


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("五子棋 20x20")
    font = load_font(22)
    train_font = load_font(11)
    small_font = load_font(18)
    clock = pygame.time.Clock()

    game = Gomoku()
    winner_text = None
    train_text = ""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = GomokuAgent(root=ROOT, device=device, simulations=80)

    mode = 0
    mode_desc = [
        "模式：人人对战",
        "模式：人执黑 vs AI执白",
        "模式：AI执黑 vs 人执白",
    ]
    mode_text = mode_desc[mode] + f" | 智能体：{agent.current_model_label}"
    if agent.model_error:
        mode_text += " | " + agent.model_error

    btn_width = 120
    btn_height = 36
    btn_spacing = 12
    btn_y = MARGIN + BOARD_PIXELS + 90
    btn_mode_rect = pygame.Rect(MARGIN, btn_y, btn_width, btn_height)
    btn_reset_rect = pygame.Rect(MARGIN + btn_width + btn_spacing, btn_y, btn_width, btn_height)
    btn_agent_rect = pygame.Rect(MARGIN + (btn_width + btn_spacing) * 2, btn_y, btn_width, btn_height)
    btn_train_rect = pygame.Rect(MARGIN + (btn_width + btn_spacing) * 3, btn_y, btn_width, btn_height)
    online_training_enabled = True

    def refresh_mode_text():
        nonlocal mode_text
        mode_text = mode_desc[mode] + f" | 智能体：{agent.current_model_label}"
        if agent.model_error:
            mode_text += " | " + agent.model_error

    def maybe_train_after_game():
        nonlocal train_text
        if online_training_enabled and mode in (1, 2) and game.finished:
            train_text = agent.finish_human_game_training(game.winner)

    def prepare_game_session():
        nonlocal train_text
        agent.begin_human_game_training()
        if mode in (1, 2) and online_training_enabled:
            train_text = "本局启用在线训练。"
        elif mode in (1, 2):
            train_text = "本局关闭在线训练。"
        else:
            train_text = ""

    def ai_move_if_needed():
        nonlocal winner_text
        if winner_text or game.finished:
            return
        if mode == 1 and game.player == WHITE:
            ai_player = game.player
            action, policy = agent.pick_move_with_policy(game)
            if online_training_enabled:
                agent.record_human_game_step(game, action, policy, ai_player)
            res = game.step(action)
            winner_text = game_result_text(res.winner, res.finished)
            if res.finished:
                maybe_train_after_game()
        elif mode == 2 and game.player == BLACK:
            ai_player = game.player
            action, policy = agent.pick_move_with_policy(game)
            if online_training_enabled:
                agent.record_human_game_step(game, action, policy, ai_player)
            res = game.step(action)
            winner_text = game_result_text(res.winner, res.finished)
            if res.finished:
                maybe_train_after_game()

    def apply_human_move_with_record(pos) -> str | None:
        action = click_to_action(pos)
        if action is None:
            return None
        if action not in set(game.legal_moves()):
            return None
        player = game.player
        if online_training_enabled and mode in (1, 2):
            agent.record_online_step(
                game=game,
                action=action,
                player=player,
                policy=None,
                is_human=True,
            )
        try:
            res = game.step(action)
        except ValueError:
            return None
        outcome = game_result_text(res.winner, res.finished)
        if res.finished:
            maybe_train_after_game()
        return outcome

    def switch_mode():
        nonlocal mode, winner_text
        mode = (mode + 1) % 3
        game.reset()
        prepare_game_session()
        winner_text = None
        refresh_mode_text()
        ai_move_if_needed()

    def reset_game():
        nonlocal winner_text
        game.reset()
        winner_text = None
        prepare_game_session()
        ai_move_if_needed()

    def switch_agent_model():
        nonlocal train_text
        agent.cycle_model()
        refresh_mode_text()
        prepare_game_session()
        if train_text:
            train_text = "已切换模型。 " + train_text
        else:
            train_text = "已切换模型。"

    def toggle_online_training():
        nonlocal online_training_enabled, train_text
        online_training_enabled = not online_training_enabled
        prepare_game_session()
        state = "开" if online_training_enabled else "关"
        train_text = f"在线训练：{state}"

    buttons = [
        Button(btn_mode_rect, "切换模式", switch_mode),
        Button(btn_reset_rect, "重置棋局", reset_game),
        Button(btn_agent_rect, "切换模型", switch_agent_model),
        Button(btn_train_rect, "训练开关", toggle_online_training),
    ]

    prepare_game_session()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for btn in buttons:
                    if btn.handle(event.pos):
                        btn.action()
                        break
                else:
                    if not winner_text:
                        if mode == 0:
                            winner_text = handle_click(game, event.pos)
                        elif mode == 1 and game.player == BLACK:
                            winner_text = apply_human_move_with_record(event.pos)
                        elif mode == 2 and game.player == WHITE:
                            winner_text = apply_human_move_with_record(event.pos)

        ai_move_if_needed()

        draw_board(screen, font, train_font, game, winner_text, mode_text, train_text)
        for btn in buttons:
            btn.draw(screen, small_font)
        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
