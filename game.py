"""Laya-Pacman: a watch-only neon maze chase.

Classic 28x31 maze, four random ghosts that mostly chase, and a Pac-Man whose turning decisions
are made only by the local Laya decision model (see laya_client.py).  No
keyboard control for Pac-Man: Esc quits, P pauses, R restarts.

Game logic is headlessly importable; ``python game.py --screenshot frame.png``
renders one styled frame without a window (SDL dummy driver).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from collections import deque

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from brain import Brain

COLS, ROWS = 28, 31
TILE = 24
HUD_TOP, HUD_BOT = 2, 2
MARGIN = 8
MAZE_W = COLS * TILE + 2 * MARGIN
PANEL_W = 380
WIDTH = MAZE_W + PANEL_W
HEIGHT = (ROWS + HUD_TOP + HUD_BOT) * TILE
MAZE_X = MARGIN
MAZE_Y = HUD_TOP * TILE
PAD = 8  # sprite padding, room for the baked glow

WALL, DOOR, PELLET, POWER = "#", "-", ".", "o"
DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
REVERSE = {"up": "down", "down": "up", "left": "right", "right": "left"}
DIR_ORDER = ("up", "down", "left", "right")

PAC_SPAWN = (13, 23)
GHOST_EXIT = (13, 11)   # tile just above the door
HOUSE_TARGET = (13, 13)
GHOST_COLORS = {"red": (255, 0, 0), "pink": (255, 184, 255),
                "cyan": (0, 255, 255), "orange": (255, 184, 81)}
GHOST_STARTS = [
    # color, spawn tile, state, release seconds
    ("red", GHOST_EXIT, "hunt", 0.0),
    ("pink", (13, 13), "house", 2.0),
    ("cyan", (12, 13), "house", 4.0),
    ("orange", (14, 13), "house", 6.0),
]

START_LIVES = 5         # lives per run
STEP_BASE = 0.110       # seconds per tile step at level 1
STEP_FLOOR = 0.070
STEP_DECAY = 0.95
CHASE_CHANCE = 0.8      # per step: hunting ghosts head toward Pac-Man, else random
STUCK_STEPS = 30        # no pellet eaten for this many moves -> lost in cleared ground
FRIGHT_BASE = 7.0       # power-pellet fright, seconds
FRIGHT_DECAY = 0.5
FRIGHT_FLOOR = 2.0
RESPAWN_WAIT = 0.5

BG = (4, 4, 10)
WALL_BLUE = (30, 107, 255)
DOOR_PINK = (255, 120, 190)
PELLET_YELLOW = (255, 232, 60)
PAC_YELLOW = (255, 255, 0)
FRIGHT_BLUE = (36, 36, 190)
EYE_WHITE = (240, 240, 245)
EYE_BLUE = (40, 80, 255)
HUD_WHITE = (225, 230, 245)
HUD_CYAN = (0, 240, 255)
HUD_DIM = (125, 138, 181)
PANEL_BG = (7, 9, 20)
PANEL_LINE = (28, 38, 80)
SOURCE_COLORS = {"laya": (255, 210, 30), "danger": (255, 90, 90),
                 "loop": (0, 240, 255), "chase": (56, 224, 138),
                 "forced": (125, 138, 181)}

LEGEND = (
    "legend: # wall . pellet o power pellet ' ' empty\n"
    "P Pac-Man (facing {facing}) G deadly ghost\n"
    "f frightened ghost (edible) E returning eyes (harmless)"
)


def load_maze(path: str = "maze.txt") -> list[str]:
    with open(path, encoding="utf-8") as f:
        rows = f.read().splitlines()
    assert len(rows) == ROWS and all(len(r) == COLS for r in rows), \
        f"{path} must be exactly {COLS}x{ROWS}"
    return rows


class Ghost:
    def __init__(self, color, x, y, state, timer):
        self.color = color
        self.x, self.y = x, y
        self.state = state          # house | leaving | hunt | eyes
        self.timer = timer
        self.fright = 0.0
        self.dir = "left"
        self.prev = None

    def __repr__(self):
        return f"<Ghost {self.color} {self.state} ({self.x},{self.y}) dir={self.dir}>"


class Game:
    """One headless-capable game.  ``decide`` is called only at junctions."""

    def __init__(self, decide=None, seed=0, maze_path="maze.txt"):
        self.maze = load_maze(maze_path)
        self.pellet_tiles = {(x, y) for y in range(ROWS) for x in range(COLS)
                             if self.maze[y][x] == PELLET}
        self.power_tiles = {(x, y) for y in range(ROWS) for x in range(COLS)
                            if self.maze[y][x] == POWER}
        self.decide = decide
        self.seed = seed
        self.rng = random.Random(seed)
        self.brain = Brain(decide)
        self.new_game()

    # ------------------------------------------------------------- setup
    def new_game(self):
        self.score = 0
        self.lives = START_LIVES
        self.level = 1
        self.game_over = False
        self.paused = False
        self.last_decision = None   # brain.last of the last junction decision
        self.last_source = None     # danger | loop | chase | laya | forced
        self.log = []
        self.pellets = set(self.pellet_tiles)
        self.power = set(self.power_tiles)
        self.steps_since_pellet = 0
        self.brain.reset()
        self._pac_dist = None
        self.reset_positions()

    def reset_positions(self):
        x, y = PAC_SPAWN
        self.pac = type("Pac", (), {"x": x, "y": y, "dir": "left", "prev": None})()
        self.steps_since_pellet = 0
        self.visited_since_pellet = {(x, y)}
        self._pac_dist = None
        self.ghosts = [Ghost(color, gx, gy, state, timer)
                       for color, (gx, gy), state, timer in GHOST_STARTS]

    @property
    def step_seconds(self) -> float:
        return max(STEP_FLOOR, STEP_BASE * STEP_DECAY ** (self.level - 1))

    @property
    def fright_seconds(self) -> float:
        return max(FRIGHT_FLOOR, FRIGHT_BASE - FRIGHT_DECAY * (self.level - 1))

    # ------------------------------------------------------------- rules
    def tile(self, x, y) -> str:
        return self.maze[y % ROWS][x % COLS]

    def passable(self, x, y, door_ok=False) -> bool:
        ch = self.tile(x, y)
        if ch == WALL:
            return False
        if ch == DOOR:
            return door_ok
        return True

    def ahead(self, x, y, direction):
        dx, dy = DIRS[direction]
        return (x + dx) % COLS, (y + dy) % ROWS

    def legal_moves(self, x, y, direction, door_ok=False):
        return [d for d in DIR_ORDER
                if d != REVERSE.get(direction)
                and self.passable(*self.ahead(x, y, d), door_ok=door_ok)]

    def bfs_next(self, sx, sy, tx, ty, door_ok=True):
        """Next tile on a shortest path, or None when already there."""
        if (sx, sy) == (tx, ty):
            return None
        prev = {(sx, sy): None}
        queue = deque([(sx, sy)])
        while queue:
            x, y = queue.popleft()
            for d in DIR_ORDER:
                nx, ny = self.ahead(x, y, d)
                if (nx, ny) in prev or not self.passable(nx, ny, door_ok=door_ok):
                    continue
                prev[(nx, ny)] = (x, y)
                if (nx, ny) == (tx, ty):
                    cur = (nx, ny)
                    while prev[cur] != (sx, sy):
                        cur = prev[cur]
                    return cur
                queue.append((nx, ny))
        raise RuntimeError(f"no path from {(sx, sy)} to {(tx, ty)}")

    def pellet_distance(self):
        """BFS steps from every walkable tile to the nearest remaining pellet."""
        dist = {}
        queue = deque()
        for tile in self.pellets | self.power:
            dist[tile] = 0
            queue.append(tile)
        while queue:
            x, y = queue.popleft()
            step = dist[(x, y)] + 1
            for dx, dy in DIRS.values():
                nx, ny = (x + dx) % COLS, (y + dy) % ROWS
                if (nx, ny) in dist or self.maze[ny][nx] in (WALL, DOOR):
                    continue
                dist[(nx, ny)] = step
                queue.append((nx, ny))
        return dist

    def distance_field(self, start):
        """BFS steps from `start` to every walkable tile (door blocked)."""
        dist = {start: 0}
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            step = dist[(x, y)] + 1
            for name in DIRS:
                nx, ny = self.ahead(x, y, name)
                if (nx, ny) in dist or self.maze[ny][nx] in (WALL, DOOR):
                    continue
                dist[(nx, ny)] = step
                queue.append((nx, ny))
        return dist

    # ------------------------------------------------------------- state
    def state_text(self, legal, note="") -> str:
        grid = [[" " if ch in (PELLET, POWER) else ch for ch in row]
                for row in self.maze]
        for x, y in self.pellets:
            grid[y][x] = PELLET
        for x, y in self.power:
            grid[y][x] = POWER
        for g in self.ghosts:
            if g.state == "eyes":
                grid[g.y][g.x] = "E"
            elif g.fright > 0:
                grid[g.y][g.x] = "f"
            else:
                grid[g.y][g.x] = "G"
        grid[self.pac.y][self.pac.x] = "P"
        for y in range(ROWS):          # Pac-Man treats the door as a wall
            for x in range(COLS):
                if grid[y][x] == DOOR:
                    grid[y][x] = WALL
        body = "\n".join("".join(r) for r in grid)
        return body + "\n" + self.instructions(legal, note)

    def instructions(self, legal, note="") -> str:
        return (LEGEND.format(facing=self.pac.dir)
                + self.ghost_danger_line() + note
                + "\nlegal moves: " + ", ".join(legal))

    def ghost_danger_line(self) -> str:
        """Tell the model where the deadly ghosts are, so its decision can avoid them."""
        deadly = {(g.x, g.y) for g in self.ghosts
                  if g.fright <= 0 and g.state not in ("eyes", "house")}
        if not deadly:
            return ""
        adjacent = [d for d in DIR_ORDER
                    if self.ahead(self.pac.x, self.pac.y, d) in deadly]
        if adjacent:
            return "\ndeadly ghost " + "/".join(adjacent) + " - do not enter!"
        return "\ndeadly ghosts hunting you - avoid them."

    # ------------------------------------------------------------- step
    def step(self):
        if self.paused or self.game_over:
            return
        dt = self.step_seconds
        for g in self.ghosts:
            if g.fright > 0:
                g.fright = max(0.0, g.fright - dt)
        self.step_pac()
        for g in self.ghosts:
            self.step_ghost(g)
        self.resolve_collisions()
        if not self.pellets and not self.power:
            self.next_level()

    def step_pac(self):
        legal = self.legal_moves(self.pac.x, self.pac.y, self.pac.dir)
        if not legal:
            raise RuntimeError(f"Pac-Man stuck at {(self.pac.x, self.pac.y)} "
                               f"dir={self.pac.dir}: maze has a dead end")
        if len(legal) >= 2:
            choice, self.last_source = self.brain.choose(self, legal)
            self.last_decision = self.brain.last
        else:
            # corridor: normally the forced move, but safety may U-turn away
            # from a deadly ghost ahead (brain.corridor_move)
            choice, self.last_source = self.brain.corridor_move(self, legal)

        allowed = set(legal)              # last line of defence: never a wall move
        reverse = REVERSE[self.pac.dir]
        if self.passable(*self.ahead(self.pac.x, self.pac.y, reverse)):
            allowed.add(reverse)          # controllers may reverse deliberately
        if choice not in allowed:
            raise RuntimeError(f"{self.last_source} returned illegal move {choice!r} "
                               f"from {sorted(allowed)}")

        self.pac.prev = (self.pac.x, self.pac.y)
        self.pac.x, self.pac.y = self.ahead(self.pac.x, self.pac.y, choice)
        self.pac.dir = choice
        self.visited_since_pellet.add((self.pac.x, self.pac.y))
        self._pac_dist = None                # ghosts chase from Pac's new tile
        self.eat()

    def step_ghost(self, g):
        if g.state == "house":
            g.prev = (g.x, g.y)              # stationary: snap, no interpolation
            g.timer -= self.step_seconds
            if g.timer <= 0:
                g.state = "leaving"
            return
        if g.state in ("leaving", "eyes"):
            target = GHOST_EXIT if g.state == "leaving" else HOUSE_TARGET
            step = self.bfs_next(g.x, g.y, *target)
            if step is None:
                g.prev = (g.x, g.y)
                if g.state == "leaving":
                    g.state = "hunt"
                else:
                    g.state = "house"
                    g.timer = RESPAWN_WAIT
                    g.fright = 0.0
                return
            g.prev = (g.x, g.y)
            g.x, g.y = step
            return
        legal = self.legal_moves(g.x, g.y, g.dir)
        if not legal:  # only possible if the maze grew a dead end
            raise RuntimeError(f"{g} has no legal move")
        if g.fright <= 0 and self.rng.random() < CHASE_CHANCE:
            choice = self.chase_move(g, legal)
        else:
            choice = self.rng.choice(legal)
        g.prev = (g.x, g.y)
        g.x, g.y = self.ahead(g.x, g.y, choice)
        g.dir = choice

    def chase_move(self, g, legal):
        """Legal move that most reduces the true path distance to Pac-Man (BFS)."""
        if self._pac_dist is None:
            self._pac_dist = self.distance_field((self.pac.x, self.pac.y))
        dist = self._pac_dist
        best, best_d = legal[0], None
        for d in legal:
            dd = dist.get(self.ahead(g.x, g.y, d), 1 << 30)
            if best_d is None or dd < best_d:
                best, best_d = d, dd
        return best

    def eat(self):
        pos = (self.pac.x, self.pac.y)
        ate = False
        if pos in self.pellets:
            self.pellets.discard(pos)
            self.score += 10
            ate = True
        if pos in self.power:
            self.power.discard(pos)
            self.score += 50
            ate = True
            for g in self.ghosts:
                if g.state == "eyes":
                    continue
                g.fright = self.fright_seconds
                if g.state == "hunt":
                    g.dir = REVERSE[g.dir]
            self.log.append(f"level {self.level}: power pellet at {pos}, "
                            f"fright {self.fright_seconds:.1f}s")
        if ate:
            self.steps_since_pellet = 0
            self.visited_since_pellet = {pos}
        else:
            self.steps_since_pellet += 1

    def resolve_collisions(self):
        px, py = self.pac.x, self.pac.y
        for g in self.ghosts:
            if g.state in ("eyes", "house"):
                continue
            same = (g.x, g.y) == (px, py)
            swapped = (g.prev == (px, py) and self.pac.prev == (g.x, g.y))
            if not (same or swapped):
                continue
            if g.fright > 0:
                self.score += 200
                g.state = "eyes"
                g.fright = 0.0
                g.prev = None
                self.log.append(f"level {self.level}: ate {g.color} ghost, "
                                f"eyes returning home")
            else:
                self.lose_life()
                return

    def lose_life(self):
        self.lives -= 1
        self.log.append(f"level {self.level}: caught, {self.lives} lives left")
        if self.lives <= 0:
            self.lives = 0
            self.game_over = True
        else:
            self.reset_positions()

    def next_level(self):
        self.level += 1
        self.pellets = set(self.pellet_tiles)
        self.power = set(self.power_tiles)
        self.reset_positions()
        self.log.append(f"level cleared -> level {self.level} "
                        f"(step {self.step_seconds * 1000:.0f} ms, "
                        f"fright {self.fright_seconds:.1f}s)")

    def restart(self):
        self.new_game()


# ----------------------------------------------------------------- rendering
def bake_glow(src: pygame.Surface, factor: int = 4) -> pygame.Surface:
    """Baked bloom: blurred copy under the sharp original.  Once at startup."""
    w, h = src.get_size()
    small = pygame.transform.smoothscale(src, (max(1, w // factor), max(1, h // factor)))
    glow = pygame.transform.smoothscale(small, (w, h))
    out = pygame.Surface((w, h), pygame.SRCALPHA)
    out.blit(glow, (0, 0))
    out.blit(glow, (0, 0), special_flags=pygame.BLEND_RGB_ADD)  # additive halo
    out.blit(src, (0, 0))
    return out


def build_wall_surface(maze) -> pygame.Surface:
    surf = pygame.Surface((MAZE_W, ROWS * TILE), pygame.SRCALPHA)
    half = max(1, int(round(TILE * 0.075)))
    lw = half * 2
    for y in range(ROWS):
        for x in range(COLS):
            if maze[y][x] != WALL:
                continue
            x0, y0 = x * TILE, y * TILE
            mid, hi_x, hi_y = half, x0 + TILE - half, y0 + TILE - half
            segments = []
            if y == 0 or maze[y - 1][x] != WALL:
                segments.append(((x0 + mid, y0 + mid), (hi_x, y0 + mid)))
            if y == ROWS - 1 or maze[y + 1][x] != WALL:
                segments.append(((x0 + mid, hi_y), (hi_x, hi_y)))
            if x == 0 or maze[y][x - 1] != WALL:
                segments.append(((x0 + mid, y0 + mid), (x0 + mid, hi_y)))
            if x == COLS - 1 or maze[y][x + 1] != WALL:
                segments.append(((hi_x, y0 + mid), (hi_x, hi_y)))
            for a, b in segments:
                pygame.draw.line(surf, WALL_BLUE, a, b, lw)
                for p in (a, b):
                    pygame.draw.circle(surf, WALL_BLUE, (int(p[0]), int(p[1])), half)
    for y in range(ROWS):          # the ghost-house gate
        for x in range(COLS):
            if maze[y][x] == DOOR:
                pygame.draw.rect(surf, DOOR_PINK,
                                 (x * TILE + 2, y * TILE + TILE // 2 - 1, TILE - 4, 2))
    return bake_glow(surf)


def blank() -> pygame.Surface:
    return pygame.Surface((TILE + 2 * PAD, TILE + 2 * PAD), pygame.SRCALPHA)


def pac_surface(direction: str, frame: int) -> pygame.Surface:
    s = blank()
    c = PAD + TILE // 2
    r = TILE // 2 - 2
    base = {"right": 0.0, "down": math.pi / 2, "left": math.pi, "up": -math.pi / 2}[direction]
    half_mouth = (0.05, 0.30, 0.55)[frame] * math.pi
    a0, a1 = base + half_mouth, base + 2 * math.pi - half_mouth
    pts = [(c, c)]
    steps = 26
    for i in range(steps + 1):
        a = a0 + (a1 - a0) * i / steps
        pts.append((c + r * math.cos(a), c + r * math.sin(a)))
    pygame.draw.polygon(s, PAC_YELLOW, pts)
    return bake_glow(s)


def ghost_surface(color, direction: str, mode: str = "normal") -> pygame.Surface:
    s = blank()
    cx = PAD + TILE // 2
    top = PAD + 2
    body_w = TILE - 6
    r = body_w // 2
    cy = top + r
    bottom = cy + 15
    body_color = FRIGHT_BLUE if mode == "fright" else color
    if mode != "eyes":
        pygame.draw.circle(s, body_color, (cx, cy), r)
        pygame.draw.rect(s, body_color, (cx - r, cy, body_w, bottom - cy))
        seg = body_w / 3.0
        skirt = [(cx - r, bottom)]
        for i in range(3):
            skirt.append((cx - r + seg * (i + 0.5), bottom + 5))
            skirt.append((cx - r + seg * (i + 1), bottom))
        pygame.draw.polygon(s, body_color, skirt)
    ex, ey = 4, cy - 1
    if mode == "fright":
        for sx in (cx - ex, cx + ex):
            pygame.draw.circle(s, EYE_WHITE, (sx, ey), 2)
    else:
        dx, dy = DIRS[direction]
        for sx in (cx - ex, cx + ex):
            pygame.draw.circle(s, EYE_WHITE, (sx, ey), 4)
            pygame.draw.circle(s, EYE_BLUE, (sx + dx * 2, ey + dy * 2), 2)
    return bake_glow(s)


def pellet_surface(power: bool) -> pygame.Surface:
    s = blank()
    c = PAD + TILE // 2
    if power:
        for i in range(10, 0, -1):
            t = i / 10.0
            col = (255, int(250 - 130 * t), int(150 - 150 * t))
            pygame.draw.circle(s, col, (c, c), i)
    else:
        pygame.draw.rect(s, PELLET_YELLOW, (c - 3, c - 3, 6, 6))
    return bake_glow(s)


class Renderer:
    def __init__(self, maze):
        self.walls = build_wall_surface(maze)
        self.pac = {d: [pac_surface(d, f) for f in range(3)] for d in DIRS}
        self.ghosts = {
            color: {d: ghost_surface(color, d) for d in DIRS}
            for color in GHOST_COLORS
        }
        self.fright = {d: ghost_surface(GHOST_COLORS["red"], d, "fright") for d in DIRS}
        self.eyes = {d: ghost_surface(GHOST_COLORS["red"], d, "eyes") for d in DIRS}
        self.pellet = pellet_surface(False)
        self.power = pellet_surface(True)
        self.font = pygame.font.Font(None, 30)
        self.big = pygame.font.Font(None, 64)
        self.tiny = pygame.font.Font(None, 19)

    @staticmethod
    def _pos(ent, frac):
        """Interpolated tile position between ent.prev and (ent.x, ent.y)."""
        if ent.prev is None or ent.prev == (ent.x, ent.y):
            return float(ent.x), float(ent.y)
        px, py = ent.prev
        if abs(ent.x - px) > 1:              # tunnel wrap: snap, don't streak
            return float(ent.x), float(ent.y)
        return px + (ent.x - px) * frac, py + (ent.y - py) * frac

    def draw(self, screen, game: Game, t: float, frac: float = 1.0, panel=None):
        screen.fill(BG)
        screen.blit(self.walls, (MAZE_X, MAZE_Y))
        for x, y in game.pellets:
            screen.blit(self.pellet, (MAZE_X + x * TILE - PAD, MAZE_Y + y * TILE - PAD))
        if int(t * 5) % 2 == 0:
            for x, y in game.power:
                screen.blit(self.power, (MAZE_X + x * TILE - PAD, MAZE_Y + y * TILE - PAD))
        for g in game.ghosts:
            if g.fright > 0 and g.fright < 2.0 and int(g.fright * 5) % 2:
                continue  # blinking during the last two seconds
            if g.state == "eyes":
                sprite = self.eyes[g.dir]
            elif g.fright > 0:
                sprite = self.fright[g.dir]
            else:
                sprite = self.ghosts[g.color][g.dir]
            gx, gy = self._pos(g, frac)
            screen.blit(sprite, (MAZE_X + gx * TILE - PAD, MAZE_Y + gy * TILE - PAD))
        frame = int(t * 14) % 3
        px, py = self._pos(game.pac, frac)
        screen.blit(self.pac[game.pac.dir][frame],
                    (MAZE_X + px * TILE - PAD, MAZE_Y + py * TILE - PAD))
        self.draw_hud(screen, game)
        self._draw_panel(screen, panel or [])

    def _draw_panel(self, screen, rows):
        """Live view of the last junction decisions (what Laya chose and why)."""
        x0 = MAZE_W
        pygame.draw.rect(screen, PANEL_BG, pygame.Rect(x0, 0, PANEL_W, HEIGHT))
        pygame.draw.line(screen, PANEL_LINE, (x0, 0), (x0, HEIGHT), 2)
        screen.blit(self.font.render("LAYA DECISIONS", True, HUD_CYAN), (x0 + 16, 10))
        screen.blit(self.tiny.render("model probabilities at each junction", True, HUD_DIM),
                    (x0 + 16, 36))
        y = 58
        if not rows:
            screen.blit(self.tiny.render("waiting for the first junction...", True, HUD_DIM),
                        (x0 + 16, y))
            return
        for rec in reversed(rows):
            color = SOURCE_COLORS.get(rec.get("source"), HUD_WHITE)
            head = (f"t{rec.get('tick', 0):<4} {rec.get('source', ''):<6} "
                    f"{rec.get('personality', ''):<10} -> {rec.get('move', '')}")
            screen.blit(self.tiny.render(head, True, color), (x0 + 16, y))
            ms = rec.get("ms")
            if ms is not None:
                info = f"{ms:.0f} ms" + ("  cached" if rec.get("cached") else "")
                screen.blit(self.tiny.render(info, True, HUD_DIM), (x0 + PANEL_W - 92, y))
            y += 16
            probs = rec.get("probs") or {}
            bx = x0 + 16
            for m in rec.get("legal") or []:
                p = probs.get(m)
                if p is None:
                    continue
                chosen = m == rec.get("move")
                col = PELLET_YELLOW if chosen else (70, 100, 190)
                screen.blit(self.tiny.render(f"{m[0].upper()} {p:.2f}", True, col), (bx, y))
                pygame.draw.rect(screen, col, (bx + 34, y + 4, max(2, int(p * 50)), 7))
                bx += 90
            if not probs:
                screen.blit(self.tiny.render("controller", True, HUD_DIM), (bx, y))
            y += 26

    def draw_hud(self, screen, game: Game):
        score = self.font.render(f"SCORE {game.score:06d}", True, HUD_WHITE)
        screen.blit(score, (MAZE_X + TILE, TILE // 2))
        level = self.font.render(f"LEVEL {game.level}", True, HUD_CYAN)
        screen.blit(level, (MAZE_W - level.get_width() - MAZE_X - TILE, TILE // 2))
        icon = pygame.transform.smoothscale(self.pac["right"][1], (TILE, TILE))
        for i in range(game.lives):
            screen.blit(icon, (MAZE_X + TILE + i * TILE, HEIGHT - TILE - MARGIN))
        if game.paused:
            self.center_text(screen, "PAUSED", HUD_CYAN)
        if game.game_over:
            self.center_text(screen, "GAME OVER - PRESS R", (255, 80, 80))

    def center_text(self, screen, text, color):
        surf = self.big.render(text, True, color)
        screen.blit(surf, ((MAZE_W - surf.get_width()) // 2,
                           (HEIGHT - surf.get_height()) // 2))


# --------------------------------------------------------------------- main
def build_client():
    from laya_client import LayaClient
    return LayaClient()


def _decision_record(game, client, last):
    """Snapshot one junction decision for the on-screen Laya panel."""
    rec = {"tick": game.brain.mem.tick, "junction": last.get("junction"),
           "move": last.get("move"), "source": last.get("source"),
           "personality": last.get("mode", ""),
           "legal": list(last.get("features", {}).keys())}
    cl = getattr(client, "last", None)
    if rec["source"] == "laya" and isinstance(cl, dict):
        rec["probs"] = dict(cl.get("probs") or {})
        rec["ms"] = cl.get("ms")
        rec["cached"] = cl.get("cached")
    return rec


def main(argv=None):
    parser = argparse.ArgumentParser(description="Laya-Pacman")
    parser.add_argument("--screenshot", metavar="PATH",
                        help="render one frame headless and save it, then exit")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the random ghosts")
    parser.add_argument("--frames", type=int, default=0,
                        help="headless smoke run: step N times as fast as possible, then exit")
    parser.add_argument("--chase", type=float, default=None,
                        help="ghost chase probability 0..1 (default 0.8)")
    parser.add_argument("--record", metavar="PATH",
                        help="record a full headless gameplay video (mp4; needs "
                             "`uv run --with imageio --with imageio-ffmpeg`)")
    parser.add_argument("--record-max-steps", type=int, default=2000,
                        help="recording cap in game steps (stops early on game over)")
    args = parser.parse_args(argv)

    global CHASE_CHANCE
    if args.chase is not None:
        CHASE_CHANCE = min(1.0, max(0.0, args.chase))

    headless = bool(args.screenshot or args.frames or args.record)
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Laya-Pacman")
    renderer = Renderer(load_maze())

    if args.screenshot:
        game = Game(decide=None, seed=args.seed)
        renderer.draw(screen, game, 0.0)
        pygame.image.save(screen, args.screenshot)
        print(f"saved {args.screenshot} ({WIDTH}x{HEIGHT})")
        pygame.quit()
        return 0

    client = build_client()
    game = Game(decide=client.decide, seed=args.seed)

    if args.record:
        try:
            import imageio.v2 as imageio
        except ImportError:
            raise SystemExit(
                "--record needs imageio: uv run --with imageio --with imageio-ffmpeg "
                "python game.py --record demo.mp4")
        panel = deque(maxlen=12)
        seen_decision = None
        t, steps = 0.0, 0
        per_step = 3                       # 3 interpolated frames per tile step
        writer = imageio.get_writer(args.record, fps=30, codec="libx264",
                                    quality=8, macro_block_size=None)
        try:
            while not game.game_over and steps < args.record_max_steps:
                game.step()
                steps += 1
                if game.last_decision is not None and game.last_decision is not seen_decision:
                    seen_decision = game.last_decision
                    panel.append(_decision_record(game, client, seen_decision))
                t += game.step_seconds
                for k in range(per_step):
                    renderer.draw(screen, game, t, k / per_step, panel)
                    writer.append_data(pygame.surfarray.array3d(screen).swapaxes(0, 1))
        finally:
            writer.close()
        print(f"recorded {args.record}: {steps} steps, "
              f"game_over={game.game_over}, score={game.score}, "
              f"level={game.level}, lives={game.lives}")
        pygame.quit()
        return 0

    if args.frames:
        for _ in range(args.frames):
            game.step()
            if game.game_over:
                break
        print(f"smoke: {args.frames} steps ok, score={game.score}, lives={game.lives}, "
              f"pellets left={len(game.pellets) + len(game.power)}")
        last = game.last_decision or {}
        print(f"smoke: last junction decision {last.get('move')} "
              f"(source {last.get('source')})")
        pygame.quit()
        return 0

    clock = pygame.time.Clock()
    accumulator = 0.0
    t = 0.0
    panel = deque(maxlen=12)
    seen_decision = None
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        t += dt
        paced = min(dt, game.step_seconds * 0.6)   # decision pauses slow the game, not the glide
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    game.paused = not game.paused
                elif event.key == pygame.K_r:
                    game.restart()
        frac = 1.0
        if not game.paused and not game.game_over:
            accumulator += paced
            if accumulator >= game.step_seconds:
                accumulator -= game.step_seconds
                if accumulator >= game.step_seconds:
                    accumulator = 0.0    # slow decision: drop the backlog, never teleport
                game.step()
            frac = min(1.0, accumulator / game.step_seconds)
        if game.last_decision is not None and game.last_decision is not seen_decision:
            seen_decision = game.last_decision
            panel.append(_decision_record(game, client, seen_decision))
        renderer.draw(screen, game, t, frac, panel)
        pygame.display.flip()
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
