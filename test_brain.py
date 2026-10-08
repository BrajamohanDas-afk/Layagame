"""Deterministic tests for brain.py (engine side). No live Laya, no GPU, no window.

    uv run python test_brain.py
"""

from __future__ import annotations

from collections import deque

import brain
from brain import (Brain, Memory, DIRS, DIR_ORDER, REVERSE, INF, ZONE_RADIUS,
                   SOURCES)
from game import COLS, DOOR, ROWS, WALL, Game, load_maze


# --------------------------------------------------------------------- helpers
def ref_bfs(maze, sources):
    """Independent reference BFS over the real maze (door blocked, wrap)."""
    dist, queue = {}, deque()
    for s in sources:
        dist[s] = 0
        queue.append(s)
    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRS.values():
            n = ((x + dx) % COLS, (y + dy) % ROWS)
            if n in dist or maze[n[1]][n[0]] in (WALL, DOOR):
                continue
            dist[n] = dist[(x, y)] + 1
            queue.append(n)
    return dist


class Obj:
    pass


class FakeGame:
    """Tiny open grid for deterministic controller/memory tests."""

    def __init__(self, w=6, h=5, pellets=(), power=()):
        self.w, self.h = w, h
        self.pellets = set(pellets)
        self.power = set(power)
        self.level, self.lives = 1, 3
        self.steps_since_pellet = 0
        self.visited_since_pellet = set()
        self.pac = Obj()
        self.pac.x, self.pac.y, self.pac.dir, self.pac.prev = 0, 0, "left", None
        self.ghosts = []

    def passable(self, x, y, door_ok=False):
        return 0 <= x < self.w and 0 <= y < self.h

    def ahead(self, x, y, direction):
        dx, dy = DIRS[direction]
        return (x + dx) % self.w, (y + dy) % self.h

    def _bfs(self, sources):
        dist, queue = {}, deque()
        for s in sources:
            dist[s] = 0
            queue.append(s)
        while queue:
            x, y = queue.popleft()
            for d in DIR_ORDER:
                n = self.ahead(x, y, d)
                if n in dist or not self.passable(*n):
                    continue
                dist[n] = dist[(x, y)] + 1
                queue.append(n)
        return dist

    def pellet_distance(self):
        return self._bfs(self.pellets | self.power)


def ghost(x, y, state="hunt", fright=0.0):
    g = Obj()
    g.x, g.y, g.state, g.fright = x, y, state, fright
    g.dir = "left"
    return g


def pac_at(game, x, y, direction):
    game.pac.x, game.pac.y, game.pac.dir = x, y, direction


# --------------------------------------------------------------------- tests
def test_constants_match_game():
    import game
    assert brain.DIRS == game.DIRS
    assert brain.REVERSE == game.REVERSE
    assert brain.DIR_ORDER == game.DIR_ORDER
    print("constants-match check ok")


def test_bfs_distance_correctness():
    game = Game(decide=None, seed=0)
    maze = load_maze()
    for source in [(1, 1), (13, 14), (6, 5)]:
        got = brain._bfs_field(game, [source])
        want = ref_bfs(maze, [source])
        assert got == want, f"field from {source} differs"
        assert got[source] == 0
        assert all(d > 0 for t, d in got.items() if t != source)
    print("bfs-distance check ok")


def test_pellet_power_and_zone_features():
    game = Game(decide=None, seed=0)
    pac_at(game, 6, 5, "up")
    legal = game.legal_moves(6, 5, "up")
    feats = Brain().features(game, legal)
    food = game.pellet_distance()
    power = brain._bfs_field(game, game.power)
    for m in legal:
        dest = game.ahead(6, 5, m)
        assert feats[m]["pellet"] == food.get(dest, INF), m
        assert feats[m]["power"] == power.get(dest, INF), m
        # zone: independent truncated BFS from the destination
        seen, queue, zone = {dest}, deque([(dest, 0)]), (1 if dest in game.pellets else 0)
        while queue:
            cur, d = queue.popleft()
            if d >= ZONE_RADIUS:
                continue
            for dm in DIR_ORDER:
                n = game.ahead(*cur, dm)
                if n in seen or not game.passable(*n):
                    continue
                seen.add(n)
                queue.append((n, d + 1))
                if n in game.pellets:
                    zone += 1
        assert feats[m]["zone"] == zone, (m, feats[m]["zone"], zone)
    print("pellet/power/zone check ok")


def test_danger_levels_and_closing():
    game = FakeGame(w=7, h=7, pellets={(6, 6)})
    pac_at(game, 3, 3, "up")
    game.ghosts = [ghost(3, 1)]              # two tiles above, deadly
    legal = ["up", "left", "right"]
    feats = Brain().features(game, legal)
    assert feats["up"]["danger"] == "high"   # d=1 + closing
    assert feats["left"]["danger"] != "high"
    assert feats["right"]["danger"] != "high"
    game.ghosts = [ghost(3, 4)]              # within 2 tiles, moving away -> still high
    feats = Brain().features(game, legal)
    assert feats["up"]["danger"] == "high"
    game.ghosts = [ghost(3, 6)]              # far and not closing -> low
    feats = Brain().features(game, legal)
    assert feats["up"]["danger"] == "low"
    game.ghosts = [ghost(3, 2)]              # ghost sits on the up tile -> high
    feats = Brain().features(game, legal)
    assert feats["up"]["danger"] == "high"
    game.ghosts = []                          # no deadly ghost -> all low
    feats = Brain().features(game, legal)
    assert all(f["danger"] == "low" for f in feats.values())
    print("danger check ok")


def test_edible_feature():
    game = FakeGame(w=7, h=7, pellets={(6, 6)})
    pac_at(game, 3, 3, "up")
    game.ghosts = [ghost(3, 1, fright=4.0)]
    feats = Brain().features(game, ["up", "right"])
    assert feats["up"]["edible"] == 1       # dest (3,2) -> ghost at (3,1)
    assert feats["right"]["edible"] == 3    # dest (4,3) -> ghost at (3,1)
    game.ghosts = [ghost(3, 1, state="eyes")]
    feats = Brain().features(game, ["up", "right"])
    assert feats["up"]["edible"] is None and feats["right"]["edible"] is None
    print("edible check ok")


def test_exploration_feature():
    game = FakeGame(w=6, h=6, pellets={(5, 5)})
    pac_at(game, 2, 2, "up")
    game.visited_since_pellet = {(2, 2), (3, 2), (2, 3)}
    feats = Brain().features(game, ["up", "right"])
    # manual count from (3,2): total reachable within radius vs visited
    seen, queue = {(3, 2)}, deque([((3, 2), 0)])
    total, fresh = 1, 0
    while queue:
        cur, d = queue.popleft()
        if d >= ZONE_RADIUS:
            continue
        for m in DIR_ORDER:
            n = game.ahead(*cur, m)
            if n in seen:
                continue
            seen.add(n)
            queue.append((n, d + 1))
            total += 1
            if n not in game.visited_since_pellet:
                fresh += 1
    want = fresh / total
    assert abs(feats["right"]["new"] - want) < 1e-9, (feats["right"]["new"], want)
    assert 0.0 <= feats["up"]["new"] <= 1.0
    print("exploration check ok")


def test_memory_visit_counts_and_decay():
    mem = Memory()
    for _ in range(9):
        mem.record(None, (1, 1), "left", "right")
    assert mem.visit_count((1, 1)) == 9
    mem.record(None, (1, 1), "left", "right")   # tick 10 -> decay 9*7//8=7, then +1
    assert mem.visit_count((1, 1)) == 8, mem.visit_count((1, 1))
    assert mem.visit_count((9, 9)) == 0
    print("memory/decay check ok")


def test_memory_resets_on_pellet_and_death():
    game = FakeGame(pellets={(5, 5)}, power={(0, 0)})
    mem = Memory()
    mem.record(game, (1, 1), "left", "right")
    assert mem.visit_count((1, 1)) == 1 and len(mem.entries) == 1
    game.pellets.clear()                          # pellet eaten -> soft reset
    mem.record(game, (2, 1), "left", "right")
    assert len(mem.entries) == 1, "pellet must clear rolling entries"
    game.level = 2                                # level change -> hard reset
    mem.record(game, (3, 1), "left", "right")
    assert mem.visit_count((3, 1)) == 1 and mem.visit_count((1, 1)) == 0
    print("memory-resets check ok")


def test_cycle_detection_abab():
    mem = Memory()
    A, B = (1, 1), (2, 1)
    assert mem.loop_info(None, A, "left") is None
    mem.record(None, A, "left", "right")
    assert mem.loop_info(None, B, "left") is None
    mem.record(None, B, "left", "right")
    assert mem.loop_info(None, A, "left") is None
    mem.record(None, A, "left", "right")
    info = mem.loop_info(None, B, "left")
    assert info and info["kind"] == "cycle" and info["period"] == 2, info
    assert info["forbid"] == (B, "right"), info
    print("A-B-A-B cycle check ok")


def test_cycle_detection_abcabc():
    mem = Memory()
    A, B, C = (1, 1), (2, 1), (3, 1)
    for j in (A, B, C, A, B):
        mem.record(None, j, "left", "right")
    info = mem.loop_info(None, C, "left")
    assert info and info["kind"] == "cycle" and info["period"] == 3, info
    assert info["forbid"] == (C, "right"), info
    print("A-B-C-A-B-C cycle check ok")


def test_repeated_tile_direction():
    mem = Memory()
    A = (4, 4)
    mem.record(None, A, "left", "right")            # 1st arrival
    assert mem.loop_info(None, A, "left") is None   # 2nd arrival: one prior
    mem.record(None, A, "left", "up")               # 2nd arrival recorded
    info = mem.loop_info(None, A, "left")           # 3rd arrival: two priors -> fire
    assert info and info["kind"] == "repeat", info
    assert info["forbid"] == (A, "up"), info        # forbids the last chosen branch
    print("repeated (tile,direction) check ok")


def test_loop_escape_and_reversal():
    game = FakeGame(w=6, h=6)
    J = (2, 2)
    mem = Memory()
    info = {"kind": "cycle", "period": 2, "forbid": (J, "up")}
    move = mem.escape_move(game, J, "right", ["up", "down"], info)
    assert move == "down", move                       # forbid up, prefer DIR_ORDER
    assert mem.is_forbidden(J, "up"), "closing branch must be forbidden"

    only_reverse = mem.escape_move(game, J, "right", ["up"], info)
    assert only_reverse == "right", only_reverse      # true U-turn allowed to escape

    unfreeze = Memory()
    unfreeze.forbidden = {(J, d): 1 << 30 for d in ("up", "down", "left", "right")}
    move = unfreeze.escape_move(game, J, "right", ["up", "down"],
                                {"kind": "repeat", "forbid": None})
    assert move in ("up", "down", "right"), move
    assert not any(unfreeze.is_forbidden(J, d) for d in ("up", "down", "right"))
    print("loop-escape check ok (forbid + reversal)")


def test_controller_priority():
    # danger beats everything and never calls the model
    def boom(*a):
        raise AssertionError("model must not be called")

    game = FakeGame(w=7, h=7, pellets={(6, 6)})
    pac_at(game, 3, 3, "up")
    game.ghosts = [ghost(3, 2)]                       # adjacent deadly
    b = Brain(boom)
    move, source = b.choose(game, ["up", "left", "right"])
    assert source == "danger" and move != "up", (move, source)
    assert game.passable(*game.ahead(3, 3, move))

    # loop beats chase
    game = FakeGame(w=7, h=7, pellets={(6, 6)})
    pac_at(game, 3, 3, "right")                       # entered_from = "left"
    game.ghosts = [ghost(3, 1, fright=4.0)]           # edible, nearby
    b = Brain(boom)
    A = (6, 6)
    b.mem.record(None, A, "left", "right")
    b.mem.record(None, (3, 3), "left", "right")
    b.mem.record(None, A, "left", "right")
    move, source = b.choose(game, ["up", "down", "right"])
    assert source == "loop", source

    # chase fires when nothing higher does
    game = FakeGame(w=7, h=7, pellets={(6, 6)})
    pac_at(game, 3, 3, "up")
    game.ghosts = [ghost(3, 1, fright=4.0)]
    b = Brain(boom)
    move, source = b.choose(game, ["up", "left", "right"])
    assert source == "chase" and move == "up", (move, source)

    # laya is the fallback
    game = FakeGame(w=7, h=7, pellets={(6, 6)})
    pac_at(game, 3, 3, "up")
    b = Brain(lambda state, ins, legal: "left")
    move, source = b.choose(game, ["up", "left", "right"])
    assert source == "laya" and move in ("up", "left", "right"), (move, source)
    assert b.last is not None and b.last["source"] == "laya"
    print("controller-priority check ok (danger > loop > chase > laya)")


def test_memory_death_hard_reset():
    game = FakeGame(pellets={(5, 5)})
    mem = Memory()
    mem.record(game, (1, 1), "left", "right")
    mem.record(game, (2, 1), "left", "right")
    assert mem.visit_count((1, 1)) == 1
    game.lives = 2                                # death -> hard reset on sync
    mem.record(game, (3, 1), "left", "right")
    assert mem.visit_count((1, 1)) == 0
    assert mem.visit_count((3, 1)) == 1
    print("death hard-reset check ok")


def test_forbidden_expiry():
    game = FakeGame(w=6, h=6)
    mem = Memory()
    J = (2, 2)
    mem.escape_move(game, J, "right", ["up", "down"],
                    {"kind": "cycle", "period": 2, "forbid": (J, "up")})
    assert mem.is_forbidden(J, "up")
    for _ in range(brain.FORBID_TICKS):
        mem.record(None, (5, 5), "left", "right")
    assert not mem.is_forbidden(J, "up"), "forbidden branch must expire"
    print("forbidden-expiry check ok")


def test_stagnation_fallback():
    game = FakeGame(w=6, h=6, pellets={(0, 5)})
    mem = Memory()
    game.steps_since_pellet = brain.STAGNATION_N
    info = mem.loop_info(game, (2, 2), "left")
    assert info and info["kind"] == "stagnation" and info["forbid"] is None, info
    game.steps_since_pellet = brain.STAGNATION_N - 1
    assert mem.loop_info(game, (2, 2), "left") is None
    print("stagnation-fallback check ok")


def test_aggregator_weights_and_fallback():
    b = Brain()
    base = {"pellet": INF, "zone": 0, "power": INF, "edible": None,
            "new": 0.0, "seen": False, "danger": "low"}
    # equal features: the confident model vote holds
    even = {"features": {"up": dict(base), "right": dict(base)}}
    assert b._aggregate(["up", "right"], even, {"up": 0.9, "right": 0.1}) == "up"
    # equal probs: the better features win
    attractive = dict(base, danger="low", pellet=1, new=1.0)
    lopsided = {"features": {"up": dict(base, danger="high", seen=True),
                             "right": attractive}}
    assert b._aggregate(["up", "right"], lopsided, {"up": 0.5, "right": 0.5}) == "right"
    # one-hot fallback: features may override the model pick (documented)
    assert b._aggregate(["up", "right"], lopsided, {"up": 1.0}) == "right"
    print("aggregator-weights check ok")


def test_dict_decide_probs():
    game = FakeGame(w=7, h=7, pellets={(6, 6)})
    pac_at(game, 3, 3, "up")

    def decide(state, instructions, legal):
        return {"move": "left", "probs": {"left": 0.1, "up": 0.8, "right": 0.1}}

    b = Brain(decide)
    move, source = b.choose(game, ["up", "left", "right"])
    assert source == "laya" and move in ("up", "left", "right"), (move, source)
    assert move == "up", "a confident model vote must survive the aggregator"
    print("dict-decide probs check ok")


def test_decision_object_probs():
    class Decision:
        move = "up"
        probs = {"up": 0.7, "left": 0.2, "right": 0.1}

    game = FakeGame(w=7, h=7, pellets={(6, 6)})
    pac_at(game, 3, 3, "up")
    b = Brain(lambda state, ins, legal: Decision())
    move, source = b.choose(game, ["up", "left", "right"])
    assert source == "laya" and move == "up", (move, source)
    print("decision-object check ok")


def test_chase_lethality():
    game = FakeGame(w=7, h=7, pellets={(6, 6)})
    pac_at(game, 3, 3, "up")
    b = Brain()
    facts = {"edible": {(3, 2): 1, (4, 3): 2},        # tiles, like real facts
             "deadly": {(3, 2): 1, (4, 3): INF},      # going up grazes a deadly ghost
             "d_pac_edible": 3, "d_pac": INF}
    assert b._chase_move(game, facts, ["up", "right"]) == "right"
    print("chase-lethality check ok")


def test_seen_staleness_after_pellet():
    game = FakeGame(w=7, h=7, pellets={(6, 6)}, power={(0, 6)})
    pac_at(game, 3, 3, "up")
    b = Brain()
    b.mem.record(game, (3, 3), "down", "up")
    assert b.features(game, ["up", "right"])["up"]["seen"] is True
    game.pellets.clear()                          # pellet eaten elsewhere
    assert b.features(game, ["up", "right"])["up"]["seen"] is False, \
        "features must sync memory before reporting seen"
    print("seen-staleness check ok")


def test_corridor_danger_escape():
    def corridor(ghost_pos=None, ghost_dir="down"):
        game = Game(decide=None, seed=0)
        game.pac.x, game.pac.y, game.pac.dir = 1, 4, "up"   # true corridor
        for g in game.ghosts:
            g.state = "eyes"
        if ghost_pos is not None:
            g = game.ghosts[0]
            g.state, g.fright, g.dir = "hunt", 0.0, ghost_dir
            g.x, g.y = ghost_pos
        legal = game.legal_moves(1, 4, "up")
        assert legal == ["up"], legal
        return game, legal

    # deadly ghost two tiles ahead -> U-turn to safety
    game, legal = corridor((1, 2), "down")
    move, source = game.brain.corridor_move(game, legal)
    assert (move, source) == ("down", "danger"), (move, source)

    # no deadly ghost -> unchanged forced move
    game, legal = corridor(None)
    assert game.brain.corridor_move(game, legal) == ("up", "forced")

    # deadly ghost chasing from behind -> keep running forward
    game, legal = corridor((1, 5), "up")
    assert game.brain.corridor_move(game, legal) == ("up", "forced")

    # deadly ghost two ahead but moving away -> no pointless U-turn
    game, legal = corridor((1, 2), "up")
    assert game.brain.corridor_move(game, legal) == ("up", "forced")

    # integration: step_pac takes the escape and tags the source
    game, _ = corridor((1, 2), "down")
    game.step_pac()
    assert (game.pac.x, game.pac.y) == (1, 5), (game.pac.x, game.pac.y)
    assert game.last_source == "danger", game.last_source
    print("corridor-danger-escape check ok")


def test_forced_move_path():
    def boom(*a):
        raise AssertionError("model must not be called on a forced move")

    game = Game(decide=boom, seed=0)
    game.pac.x, game.pac.y, game.pac.dir = 1, 14, "left"     # tunnel corridor
    game.step_pac()
    assert game.last_source == "forced", game.last_source
    print("forced-move check ok")


def test_game_guard_direct():
    game = Game(decide=None, seed=0)
    game.pac.x, game.pac.y, game.pac.dir = 1, 3, "left"      # left is a wall
    game.brain.choose = lambda g, l: ("left", "laya")
    try:
        game.step_pac()
    except RuntimeError as exc:
        assert "illegal" in str(exc)
    else:
        raise AssertionError("step_pac guard did not reject a wall move")

    game = Game(decide=None, seed=0)
    game.pac.x, game.pac.y, game.pac.dir = 6, 5, "right"
    game.brain.choose = lambda g, l: ("left", "danger")      # deliberate U-turn
    game.step_pac()
    assert (game.pac.x, game.pac.y) == (5, 5) and game.last_source == "danger"
    print("game-guard check ok (wall rejected, controller U-turn allowed)")


def test_illegal_move_protection():
    def bad(state, ins, legal):
        return "left"                                # reverse of Pac's direction

    game = Game(decide=bad, seed=0)
    game.pac.x, game.pac.y, game.pac.dir = 6, 5, "right"
    legal = game.legal_moves(6, 5, "right")
    assert "left" not in legal
    try:
        game.step_pac()
    except RuntimeError as exc:
        assert "illegal" in str(exc)
    else:
        raise AssertionError("illegal move was not rejected")
    print("illegal-move-protection check ok")


def test_no_wall_entry_simulation():
    import random

    for seed in range(4):
        rng = random.Random(seed)
        game = Game(decide=lambda s, i, l: rng.choice(l), seed=seed)
        for _ in range(300):
            game.step()
            assert game.tile(game.pac.x, game.pac.y) != WALL
            assert game.last_source in SOURCES, game.last_source
            for g in game.ghosts:
                assert game.tile(g.x, g.y) != WALL
    print("no-wall simulation check ok (4 seeds x 300 ticks)")


def test_prompt_has_no_board():
    captured = {}

    def decide(state, instructions, legal):
        captured["state"] = state
        return legal[0]

    game = Game(decide=decide, seed=0)
    game.pac.x, game.pac.y, game.pac.dir = 6, 5, "up"
    game.step_pac()
    state = captured["state"]
    assert "#" not in state, "features-only prompt must not contain the board"
    assert state.count("\n") <= 8, state
    assert len(state) < 600, len(state)
    print("no-board-prompt check ok")


def test_prompt_matches_client_formatter():
    from laya_client import format_features_state

    game = Game(decide=None, seed=0)
    pac_at(game, 6, 5, "up")
    legal = game.legal_moves(6, 5, "up")
    b = Brain()
    facts = b._facts(game, legal)
    state = b.prompt(game, legal, facts)
    want, _instructions, _mapping = format_features_state(
        facts["features"], b._meta(game, facts), "labels")
    assert state == want, "brain.prompt must delegate to the client formatter"
    print("prompt-formatter-unified check ok")


def test_features_contract():
    game = Game(decide=None, seed=0)
    pac_at(game, 6, 5, "up")
    legal = game.legal_moves(6, 5, "up")
    feats = Brain().features(game, legal)
    assert set(feats) == set(legal)
    for m, f in feats.items():
        assert set(f) == {"pellet", "zone", "power", "danger", "edible", "new", "seen"}, (m, sorted(f))
        assert f["danger"] in ("low", "med", "high")
        assert isinstance(f["pellet"], int) and isinstance(f["zone"], int)
        assert isinstance(f["new"], float) and 0.0 <= f["new"] <= 1.0
        assert isinstance(f["seen"], bool)
    print("feature-contract check ok")


def test_seen_feature():
    game = Game(decide=None, seed=0)
    pac_at(game, 6, 5, "up")
    legal = game.legal_moves(6, 5, "up")
    b = Brain()
    b.mem.record(None, (6, 5), "down", "up")          # entered_from for dir=up
    feats = b.features(game, legal)
    assert feats["up"]["seen"] is True
    assert all(feats[m]["seen"] is False for m in legal if m != "up")
    print("seen-feature check ok")


def test_danger_side_approach():
    """A ghost approaching from the side must trigger danger even when Pac's
    current moves do not close the gap (the old check missed these)."""
    def boom(*a):
        raise AssertionError("model must not be called")

    game = FakeGame(w=7, h=7, pellets={(6, 6)})
    pac_at(game, 3, 3, "up")
    g = ghost(0, 3)                      # 3 tiles left, moving right toward Pac
    g.dir = "right"
    game.ghosts = [g]
    b = Brain(boom)
    move, source = b.choose(game, ["up", "left", "right"])
    assert source == "danger", (move, source)
    assert move != "left", (move, source)   # must not walk toward the ghost
    print("danger side-approach check ok")


def main():
    test_constants_match_game()
    test_bfs_distance_correctness()
    test_pellet_power_and_zone_features()
    test_danger_levels_and_closing()
    test_danger_side_approach()
    test_edible_feature()
    test_exploration_feature()
    test_memory_visit_counts_and_decay()
    test_memory_resets_on_pellet_and_death()
    test_memory_death_hard_reset()
    test_forbidden_expiry()
    test_stagnation_fallback()
    test_cycle_detection_abab()
    test_cycle_detection_abcabc()
    test_repeated_tile_direction()
    test_loop_escape_and_reversal()
    test_controller_priority()
    test_aggregator_weights_and_fallback()
    test_dict_decide_probs()
    test_decision_object_probs()
    test_chase_lethality()
    test_seen_staleness_after_pellet()
    test_corridor_danger_escape()
    test_forced_move_path()
    test_game_guard_direct()
    test_illegal_move_protection()
    test_no_wall_entry_simulation()
    test_prompt_has_no_board()
    test_prompt_matches_client_formatter()
    test_features_contract()
    test_seen_feature()
    print("all brain tests passed")


if __name__ == "__main__":
    main()
