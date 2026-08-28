"""Play against the trained network in a window (Arcade).

The same weights the trainer writes are used here: whatever `make train`
produced under ``runs/<name>/checkpoints`` is loaded through the usual backend
selection, so this runs on CUDA, on Apple MLX, on Metal via PyTorch MPS, or on
the CPU without any change to this file.

Run it with ``make gui`` (see ``omok gui --help`` for the options).  The search
lives on a worker thread -- see :mod:`omok.engine` -- and this module only
draws and collects input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import arcade
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError(
        "the Arcade GUI needs the `arcade` package -- run `make install-gui`"
    ) from exc

from .board import BLACK, Board, DIRECTIONS, EMPTY, WHITE, board_from_moves, format_move
from .config import Config
from .engine import Engine

# ---------------------------------------------------------------- palette
WOOD = (216, 178, 126)
WOOD_EDGE = (166, 128, 80)
GRID = (64, 48, 34)
BLACK_STONE = (26, 26, 30)
BLACK_SHEEN = (72, 74, 82)
WHITE_STONE = (243, 241, 236)
WHITE_SHEEN = (255, 255, 255)
STONE_EDGE = (38, 38, 42)
PANEL_BG = (23, 25, 30)
PANEL_RULE = (46, 50, 59)
TEXT = (228, 232, 240)
TEXT_DIM = (136, 145, 160)
ACCENT = (110, 175, 255)
GOOD = (122, 202, 142)
WARN = (232, 178, 92)
MONO = ("Menlo", "DejaVu Sans Mono", "Consolas", "monospace")

# Difficulty presets bound to the number keys.
LEVELS = ((1, "beginner", 24), (2, "casual", 100), (3, "strong", 320), (4, "max", 900))
# The four ways the two colours can be shared between human and engine.
MODES = ("black", "white", "none", "both")
MODE_LABELS = {"black": "you play black", "white": "you play white",
               "none": "engine vs engine", "both": "two humans"}


def human_colours(mode: str) -> frozenset[int]:
    return {
        "black": frozenset({BLACK}),
        "white": frozenset({WHITE}),
        "none": frozenset(),
        "both": frozenset({BLACK, WHITE}),
    }[mode]


# --------------------------------------------------------------- geometry
@dataclass(frozen=True)
class Geometry:
    """Maps board coordinates to pixels.  Row 0 is the top row, as in the ASCII
    board and in ``format_move``, so 'h7' means the same thing everywhere."""

    size: int
    cell: float
    margin: float
    left: float = 0.0
    bottom: float = 0.0

    @property
    def extent(self) -> float:
        return self.cell * (self.size - 1) + 2 * self.margin

    @classmethod
    def fit(cls, size: int, left: float, bottom: float, width: float,
            height: float) -> "Geometry":
        """Largest board that fits the given area, centred inside it."""
        side = max(80.0, min(width, height))
        cell = side / (size - 1 + 1.6)  # 0.8 cell of margin on each side
        geo = cls(size, cell, cell * 0.8)
        return cls(size, cell, cell * 0.8,
                   left + (width - geo.extent) / 2.0,
                   bottom + (height - geo.extent) / 2.0)

    def point(self, row: int, col: int) -> tuple[float, float]:
        return (self.left + self.margin + col * self.cell,
                self.bottom + self.extent - self.margin - row * self.cell)

    def point_of(self, index: int) -> tuple[float, float]:
        return self.point(*divmod(index, self.size))

    def hit(self, x: float, y: float) -> int | None:
        """Board index under the cursor, or None when it is off the grid."""
        col = round((x - self.left - self.margin) / self.cell)
        row = round((self.bottom + self.extent - self.margin - y) / self.cell)
        if not (0 <= row < self.size and 0 <= col < self.size):
            return None
        px, py = self.point(row, col)
        if math.hypot(x - px, y - py) > self.cell * 0.5:
            return None
        return row * self.size + col

    def star_points(self) -> list[tuple[int, int]]:
        n = self.size
        if n >= 13:
            offsets = (3, n // 2, n - 4)
        elif n >= 9:
            offsets = (2, n // 2, n - 3)
        else:
            return [(n // 2, n // 2)]
        return [(r, c) for r in offsets for c in offsets]


def winning_line(board: Board) -> list[int]:
    """The stones that ended the game, so they can be highlighted."""
    if not board.over or board.winner == 0 or not board.moves:
        return []
    last = board.moves[-1]
    size, cells, player = board.size, board.cells, board.winner
    row, col = divmod(last, size)
    for dr, dc in DIRECTIONS:
        line = [last]
        for step in (1, -1):
            r, c = row + dr * step, col + dc * step
            while 0 <= r < size and 0 <= c < size and cells[r * size + c] == player:
                line.append(r * size + c)
                r += dr * step
                c += dc * step
        if len(line) >= board.win_length:
            return sorted(line)
    return []


class TextCache:
    """Reuses ``arcade.Text`` objects -- building them every frame is costly."""

    def __init__(self) -> None:
        self._items: dict[str, arcade.Text] = {}

    def draw(self, key: str, text: str, x: float, y: float, color=TEXT,
             size: float = 12.0, **kwargs) -> None:
        item = self._items.get(key)
        if item is None:
            item = arcade.Text(text, x, y, color, size, **kwargs)
            self._items[key] = item
        else:
            if item.text != text:
                item.text = text
            item.x, item.y = x, y
            item.color = color
        item.draw()


# ----------------------------------------------------------------- window
class OmokWindow(arcade.Window):
    """The board, the side panel, and the glue to the engine thread."""

    def __init__(self, cfg: Config, model_path: str | None = None,
                 simulations: int | None = None, mode: str = "black",
                 opening_plies: int = 2, opening_temperature: float = 0.8) -> None:
        size = cfg.game.board_size
        cell = max(26.0, min(46.0, 660.0 / size))
        board_px = cell * (size - 1 + 1.6)
        panel = 330.0
        super().__init__(int(board_px + panel), int(max(board_px, 620)),
                         f"Omok - {cfg.run_name}", resizable=True)
        self.background_color = PANEL_BG

        self.cfg = cfg
        self.board_size = size
        self.simulations = int(simulations or cfg.mcts.simulations)
        self.mode = mode if mode in MODES else "black"
        self.humans = human_colours(self.mode)
        self.opening_plies = opening_plies
        self.opening_temperature = opening_temperature

        self.board = self._new_board()
        self.panel_width = panel
        self.geometry = Geometry.fit(size, 0, 0, self.width - panel, self.height)

        self.engine = Engine(cfg, model_path)
        self.engine.start()
        self.ready = False
        self.info: dict = {}
        self.status = "loading model..."
        self.status_color = TEXT_DIM
        self.paused = False

        self.pending: tuple[int, str] | None = None   # (job, kind)
        self.progress = (0, 0)
        self.hover: int | None = None
        self.hints: list[tuple[int, float]] = []
        self.hints_ply = -1
        self.show_hints = True
        self.eval_black: float | None = None
        self.last_search: str = ""
        self.text = TextCache()

    @property
    def panel_left(self) -> float:
        return self.width - self.panel_width

    # -- state -------------------------------------------------------------
    def _new_board(self) -> Board:
        return Board(self.cfg.game.board_size, self.cfg.game.win_length,
                     self.cfg.game.allow_overline)

    def _reset(self, status: str | None = None) -> None:
        self.engine.cancel()
        self.pending = None
        self.progress = (0, 0)
        self.hints, self.hints_ply = [], -1
        self.eval_black = None
        self.last_search = ""
        if status:
            self.status, self.status_color = status, TEXT_DIM

    def _side_name(self, colour: int) -> str:
        return "black" if colour == BLACK else "white"

    def _engine_turn(self) -> bool:
        return not self.board.over and self.board.to_move not in self.humans

    def _play(self, move: int) -> None:
        self.board.play(move)
        if self.board.over:
            self.engine.cancel()
            self.pending = None
            if self.board.winner == EMPTY:
                self.status, self.status_color = "draw", TEXT_DIM
            else:
                self.status = f"{self._side_name(self.board.winner)} wins"
                self.status_color = GOOD

    def _submit(self, kind: str) -> None:
        temperature = (self.opening_temperature
                       if kind == "move" and self.board.move_number < self.opening_plies
                       else 0.0)
        job = self.engine.submit(self.board, self.simulations, temperature, kind)
        self.pending = (job, kind)
        self.progress = (0, self.simulations)

    # -- arcade callbacks --------------------------------------------------
    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        self.panel_width = max(300.0, min(380.0, width * 0.33))
        self.geometry = Geometry.fit(self.board_size, 0, 0,
                                     width - self.panel_width, height)

    def on_update(self, delta_time: float) -> None:
        for event in self.engine.poll():
            self._handle(event)
        if (self.ready and not self.paused and self.pending is None
                and self._engine_turn()):
            self._submit("move")

    def _handle(self, event: tuple) -> None:
        kind = event[0]
        if kind == "ready":
            self.ready = True
            self.info = event[1]
            self.status, self.status_color = "your move", TEXT_DIM
        elif kind == "error":
            self.status, self.status_color = event[1], WARN
        elif kind == "progress":
            job, done, total = event[1], event[2], event[3]
            if self.pending and job == self.pending[0]:
                self.progress = (done, total)
        elif kind == "result":
            result = event[1]
            if not self.pending or result.job != self.pending[0]:
                return  # a search we already abandoned
            self.pending = None
            sign = 1.0 if self.board.to_move == BLACK else -1.0
            self.eval_black = sign * result.value
            self.hints, self.hints_ply = result.top, result.ply
            self.last_search = (f"{result.simulations} sims in {result.seconds:.1f}s "
                                f"({result.nps:.0f}/s)")
            if result.kind == "move":
                self._play(result.move)
                if not self.board.over:
                    self.status = f"{self._side_name(self.board.to_move)} to play"
                    self.status_color = TEXT_DIM
            else:
                self.status, self.status_color = "hint ready", ACCENT

    def on_mouse_motion(self, x: float, y: float, dx: float, dy: float) -> None:
        self.hover = self.geometry.hit(x, y)

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int) -> None:
        if button != arcade.MOUSE_BUTTON_LEFT or self.board.over:
            return
        if self.board.to_move not in self.humans:
            self.status, self.status_color = "not your turn", WARN
            return
        move = self.geometry.hit(x, y)
        if move is None:
            return
        if not self.board.is_legal(move):
            self.status, self.status_color = "occupied", WARN
            return
        if self.pending is not None:  # a hint was still running
            self.engine.cancel()
            self.pending = None
        self._play(move)
        if not self.board.over:
            self.status = f"{self._side_name(self.board.to_move)} to play"
            self.status_color = TEXT_DIM

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        key = arcade.key
        if symbol in (key.ESCAPE, key.Q):
            self.engine.shutdown()
            self.close()
        elif symbol == key.N:
            self.board = self._new_board()
            self._reset("new game")
        elif symbol == key.U:
            self._undo()
        elif symbol == key.SPACE:
            if self.ready and not self.board.over and self.pending is None:
                self._submit("move")
        elif symbol == key.H:
            if self.ready and not self.board.over and self.pending is None:
                self._submit("hint")
                self.status, self.status_color = "thinking...", ACCENT
        elif symbol == key.A:
            self.show_hints = not self.show_hints
        elif symbol == key.P:
            self.paused = not self.paused
            if self.paused:
                self.engine.cancel()
                self.pending = None
            self.status = "paused" if self.paused else "running"
            self.status_color = WARN if self.paused else TEXT_DIM
        elif symbol == key.S:
            self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
            self.humans = human_colours(self.mode)
            self.engine.cancel()
            self.pending = None
            self.status, self.status_color = MODE_LABELS[self.mode], TEXT_DIM
        elif symbol in (key.KEY_1, key.KEY_2, key.KEY_3, key.KEY_4):
            level = LEVELS[symbol - key.KEY_1]
            self.simulations = level[2]
            self.engine.cancel()
            self.pending = None
            self.status = f"{level[1]}: {self.simulations} sims/move"
            self.status_color = TEXT_DIM

    def _undo(self) -> None:
        """Take back to the previous position the human is to move in."""
        moves = list(self.board.moves)
        if not moves:
            return
        take = 1
        if self.humans and self.board.to_move in self.humans and len(moves) >= 2:
            take = 2  # undo the engine's reply as well as our own move
        self.board = board_from_moves(moves[:-take], self.board_size,
                                      self.cfg.game.win_length,
                                      self.cfg.game.allow_overline)
        self._reset(f"took back {take} move{'s' if take > 1 else ''}")

    # -- drawing -----------------------------------------------------------
    def on_draw(self) -> None:
        self.clear()
        self._draw_board()
        self._draw_hints()
        self._draw_stones()
        self._draw_hover()
        self._draw_panel()

    def _draw_board(self) -> None:
        geo = self.geometry
        arcade.draw_lrbt_rectangle_filled(geo.left, geo.left + geo.extent, geo.bottom,
                                          geo.bottom + geo.extent, WOOD)
        arcade.draw_lrbt_rectangle_outline(geo.left, geo.left + geo.extent, geo.bottom,
                                           geo.bottom + geo.extent, WOOD_EDGE, 2)
        first_x, first_y = geo.point(self.board_size - 1, 0)
        last_x, last_y = geo.point(0, self.board_size - 1)
        for i in range(self.board_size):
            _, y = geo.point(i, 0)          # row i
            x, _ = geo.point(0, i)          # column i
            arcade.draw_line(first_x, y, last_x, y, GRID, 1.2)
            arcade.draw_line(x, first_y, x, last_y, GRID, 1.2)
        for row, col in geo.star_points():
            x, y = geo.point(row, col)
            arcade.draw_circle_filled(x, y, max(2.0, geo.cell * 0.09), GRID)
        # Coordinates: letters under the columns, numbers left of the rows.
        for i in range(self.board_size):
            x, _ = geo.point(0, i)
            _, y = geo.point(i, 0)
            self.text.draw(f"col{i}", chr(ord("a") + i), x, first_y - geo.margin * 0.78,
                           GRID, geo.cell * 0.32, anchor_x="center", font_name=MONO)
            self.text.draw(f"row{i}", str(i), first_x - geo.margin * 0.45, y, GRID,
                           geo.cell * 0.32, anchor_x="right", anchor_y="center",
                           font_name=MONO)

    def _draw_hints(self) -> None:
        """Translucent discs on the engine's candidate moves."""
        if not (self.show_hints and self.hints) or self.hints_ply != self.board.move_number:
            return
        geo = self.geometry
        best = max(p for _, p in self.hints)
        for rank, (move, prob) in enumerate(self.hints):
            x, y = geo.point_of(move)
            radius = geo.cell * (0.16 + 0.26 * (prob / best) ** 0.5)
            alpha = int(70 + 130 * (prob / best))
            arcade.draw_circle_filled(x, y, radius, (*ACCENT, alpha))
            if rank == 0:
                self.text.draw("hintpct", f"{prob * 100:.0f}", x, y - geo.cell * 0.13,
                               (12, 20, 32), geo.cell * 0.3, anchor_x="center",
                               font_name=MONO)

    def _draw_stones(self) -> None:
        geo = self.geometry
        radius = geo.cell * 0.44
        cells = self.board.cells
        last = self.board.moves[-1] if self.board.moves else None
        win = set(winning_line(self.board))
        for index in range(len(cells)):
            colour = cells[index]
            if colour == EMPTY:
                continue
            x, y = geo.point_of(index)
            body = BLACK_STONE if colour == BLACK else WHITE_STONE
            arcade.draw_circle_filled(x, y, radius, body)
            arcade.draw_circle_outline(x, y, radius, STONE_EDGE, 1.2)
            sheen = BLACK_SHEEN if colour == BLACK else WHITE_SHEEN
            arcade.draw_circle_filled(x - radius * 0.3, y + radius * 0.3,
                                      radius * 0.22, sheen)
            if index in win:
                arcade.draw_circle_outline(x, y, radius * 0.92, GOOD, 2.5)
            elif index == last:
                mark = WARN if colour == BLACK else (200, 90, 60)
                arcade.draw_circle_filled(x, y, radius * 0.18, mark)

    def _draw_hover(self) -> None:
        if (self.hover is None or self.board.over
                or self.board.to_move not in self.humans
                or not self.board.is_legal(self.hover)):
            return
        x, y = self.geometry.point_of(self.hover)
        body = BLACK_STONE if self.board.to_move == BLACK else WHITE_STONE
        arcade.draw_circle_filled(x, y, self.geometry.cell * 0.44, (*body, 110))

    # -- panel -------------------------------------------------------------
    def _draw_panel(self) -> None:
        left, right = self.panel_left, self.width
        arcade.draw_lrbt_rectangle_filled(left, right, 0, self.height, PANEL_BG)
        arcade.draw_line(left, 0, left, self.height, PANEL_RULE, 1)
        x = left + 20
        y = self.height - 34
        step = 18

        self.text.draw("title", "OMOK", x, y, TEXT, 20, bold=True)
        y -= 26
        self.text.draw("run", self.cfg.run_name, x, y, TEXT_DIM, 11, font_name=MONO)
        y -= step

        spec = self.info.get("spec")
        model = str(self.info.get("source", "loading..."))
        if len(model) > 34:
            model = "..." + model[-31:]
        self.text.draw("model", model, x, y, TEXT_DIM, 10, font_name=MONO)
        y -= step
        if spec is not None:
            self.text.draw(
                "net",
                f"{self.info.get('backend', '?')}/{self.info.get('device', '?')}  "
                f"{spec.blocks}x{spec.channels}  {spec.parameter_count() / 1e6:.2f}M",
                x, y, TEXT_DIM, 10, font_name=MONO)
        y -= step + 8

        y = self._draw_rule(x, y, right)
        self.text.draw("mode", MODE_LABELS[self.mode], x, y, TEXT, 12)
        y -= step
        level = next((name for _, name, sims in LEVELS if sims == self.simulations), "custom")
        self.text.draw("sims", f"{self.simulations} sims/move ({level})", x, y,
                       TEXT_DIM, 11, font_name=MONO)
        y -= step + 10

        y = self._draw_eval(x, y, right)
        y = self._draw_thinking(x, y, right)

        y = self._draw_rule(x, y, right)
        turn = ("game over" if self.board.over
                else f"move {self.board.move_number + 1}, "
                     f"{self._side_name(self.board.to_move)} to play")
        self.text.draw("turn", turn, x, y, TEXT, 12)
        y -= step
        self.text.draw("status", self.status, x, y, self.status_color, 11)
        y -= step
        if self.last_search:
            self.text.draw("speed", self.last_search, x, y, TEXT_DIM, 10, font_name=MONO)
        y -= step + 6

        y = self._draw_top_moves(x, y, right)
        y = self._draw_history(x, y)
        self._draw_keys(x)

    def _draw_rule(self, x: float, y: float, right: float) -> float:
        arcade.draw_line(x, y + 14, right - 20, y + 14, PANEL_RULE, 1)
        return y

    def _draw_eval(self, x: float, y: float, right: float) -> float:
        width = right - 20 - x
        arcade.draw_lrbt_rectangle_filled(x, x + width, y - 2, y + 14, (52, 56, 66))
        if self.eval_black is not None:
            share = (max(-1.0, min(1.0, self.eval_black)) + 1.0) / 2.0
            arcade.draw_lrbt_rectangle_filled(x, x + width * share, y - 2, y + 14,
                                              (18, 18, 22))
            arcade.draw_lrbt_rectangle_filled(x + width * share, x + width, y - 2, y + 14,
                                              (222, 220, 214))
            leader = "black" if self.eval_black >= 0 else "white"
            label = f"{leader} {abs(self.eval_black):+.2f}"
        else:
            label = "no evaluation yet"
        arcade.draw_lrbt_rectangle_outline(x, x + width, y - 2, y + 14, PANEL_RULE, 1)
        self.text.draw("evallabel", label, x, y - 20, TEXT_DIM, 10, font_name=MONO)
        return y - 44

    def _draw_thinking(self, x: float, y: float, right: float) -> float:
        if self.pending is None:
            return y
        done, total = self.progress
        width = right - 20 - x
        frac = done / total if total else 0.0
        arcade.draw_lrbt_rectangle_filled(x, x + width, y, y + 6, (44, 48, 58))
        arcade.draw_lrbt_rectangle_filled(x, x + width * frac, y, y + 6, ACCENT)
        kind = "thinking" if self.pending[1] == "move" else "analysing"
        self.text.draw("think", f"{kind} {done}/{total}", x, y - 16, ACCENT, 10,
                       font_name=MONO)
        return y - 34

    def _draw_top_moves(self, x: float, y: float, right: float) -> float:
        if not self.hints:
            return y
        stale = self.hints_ply != self.board.move_number
        self.text.draw("candhead", "candidates" + (" (previous move)" if stale else ""),
                       x, y, TEXT_DIM, 10)
        y -= 17
        width = right - 20 - x
        for rank, (move, prob) in enumerate(self.hints[:5]):
            colour = ACCENT if rank == 0 and not stale else TEXT_DIM
            arcade.draw_lrbt_rectangle_filled(x, x + width * prob, y - 3, y + 11,
                                              (40, 60, 84))
            self.text.draw(f"cand{rank}", f"{format_move(move, self.board_size):<4}"
                                          f"{prob * 100:5.1f}%",
                           x + 4, y, colour, 11, font_name=MONO)
            y -= 17
        return y - 8

    def _draw_history(self, x: float, y: float) -> float:
        moves = self.board.moves
        if not moves:
            return y
        self.text.draw("histhead", "moves", x, y, TEXT_DIM, 10)
        y -= 16
        recent = moves[-8:]
        start = len(moves) - len(recent) + 1
        for row in range(0, len(recent), 4):
            chunk = recent[row:row + 4]
            line = "  ".join(f"{start + row + i:>3}.{format_move(m, self.board_size):<4}"
                             for i, m in enumerate(chunk))
            self.text.draw(f"hist{row}", line, x, y, TEXT_DIM, 10, font_name=MONO)
            y -= 15
        return y

    def _draw_keys(self, x: float) -> None:
        lines = ("click  place a stone", "space  engine moves now",
                 "h  hint      a  overlay", "u  undo      n  new game",
                 "s  swap sides   p  pause", "1-4  difficulty   q  quit")
        y = 18 + 14 * (len(lines) - 1)
        for i, line in enumerate(lines):
            self.text.draw(f"key{i}", line, x, y - i * 14, (96, 104, 118), 10,
                           font_name=MONO)


def run_gui(cfg: Config, model_path: str | None = None, simulations: int | None = None,
            human: str = "black", opening_plies: int = 2) -> None:  # pragma: no cover
    window = OmokWindow(cfg, model_path=model_path, simulations=simulations,
                        mode=human, opening_plies=opening_plies)
    try:
        arcade.run()
    finally:
        window.engine.shutdown()
