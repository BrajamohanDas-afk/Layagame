"""Logic and integration checks for Laya-Pacman.

  uv run python check.py          stub simulation + scenario checks, no Laya
  uv run python check.py --live   GPU smoke: load Laya on CUDA, one real decision
"""

from __future__ import annotations

import argparse
import random
import sys

from game import (COLS, DIRS, DOOR, PAC_SPAWN, REVERSE, ROWS, START_LIVES,
                  WALL, Game, load_maze)


class StubClient:
    """Deterministic random stand-in for Laya: uniform over legal moves."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.calls = 0

    def decide(self, state: str, instructions: str, legal: tuple[str, ...]) -> str:
        self.calls += 1
        return self.rng.choice(legal)


def maze_walkable(x: int, y: int) -> bool:
    return load_maze()[y % ROWS][x % COLS] != WALL


def check_simulations(seeds=range(8), ticks=500):
    """500-tick headless runs x several seeds, invariants asserted every tick."""
    for seed in seeds:
        stub = StubClient(seed)
        game = Game(decide=stub.decide, seed=seed)
        prev_count = len(game.pellets) + len(game.power)
        for tick in range(ticks):
            lives0, level0 = game.lives, game.level
            pac0 = (game.pac.x, game.pac.y, game.pac.dir)
            ghosts0 = [(g.x, g.y, g.dir, g.state, g.fright) for g in game.ghosts]
            game.step()

            assert maze_walkable(game.pac.x, game.pac.y), (seed, tick, "pac in wall")
            for g in game.ghosts:
                assert maze_walkable(g.x, g.y), (seed, tick, f"{g.color} in wall")
                if g.state == "hunt":
                    assert game.tile(g.x, g.y) != DOOR, (seed, tick, "ghost in door")
            count = len(game.pellets) + len(game.power)
            assert count <= prev_count, (seed, tick, "pellets grew")
            prev_count = count

            if game.lives == lives0 and game.level == level0:
                if (game.pac.x, game.pac.y) != pac0[:2]:
                    # only the danger/loop controllers may deliberately U-turn
                    assert game.pac.dir != REVERSE[pac0[2]] or \
                        game.last_source in ("danger", "loop"), \
                        (seed, tick, "pac reversed", game.last_source)
                    dist = abs(game.pac.x - pac0[0]) + abs(game.pac.y - pac0[1])
                    assert dist == 1 or dist == COLS - 1, (seed, tick, "pac jumped")
                for (gx, gy, gd, gs, gf), g in zip(ghosts0, game.ghosts):
                    if (g.x, g.y) != (gx, gy):
                        dist = abs(g.x - gx) + abs(g.y - gy)
                        assert dist == 1 or dist == COLS - 1, (seed, tick, "ghost jumped")
        print(f"  seed {seed}: {ticks} ticks ok "
              f"(score={game.score}, level={game.level}, lives={game.lives})")
    print("simulation checks ok")


def check_level_clears():
    game = Game(decide=StubClient(0).decide, seed=0)
    game.step()  # take one step first so the state is mid-game
    level = game.level
    game.pellets.clear()
    game.power.clear()
    game.step()
    assert game.level == level + 1, "clearing all pellets must advance the level"
    assert len(game.pellets) == len(game.pellet_tiles), "next level refills every dot"
    assert len(game.power) == len(game.power_tiles), "next level refills every energizer"
    print("level-clear check ok")


def check_power_pellet():
    game = Game(decide=StubClient(1).decide, seed=1)
    game.pac.x, game.pac.y = 1, 3          # a corner power pellet
    for g in game.ghosts:
        g.state, g.fright, g.dir = "hunt", 0.0, "left"
    game.eat()
    assert game.score == 50, game.score
    for g in game.ghosts:
        assert g.fright > 0, f"{g.color} not frightened"
        assert g.dir == "right", f"{g.color} did not reverse"
    assert (1, 3) not in game.power
    print("power-pellet check ok (all ghosts reverse + frightened)")


def check_eat_ghost_and_eyes_home():
    game = Game(decide=StubClient(2).decide, seed=2)
    ghost = game.ghosts[0]
    ghost.state, ghost.fright = "hunt", 3.0
    ghost.x, ghost.y = game.pac.x, game.pac.y
    game.resolve_collisions()
    assert ghost.state == "eyes" and ghost.fright == 0
    assert game.score == 200, game.score

    ghost.x, ghost.y = 1, 29              # far corner -> BFS home
    for _ in range(200):
        game.step_ghost(ghost)
        if ghost.state == "house":
            break
    assert ghost.state == "house", f"eyes never made it home: {ghost}"
    for _ in range(60):
        game.step_ghost(ghost)
        if ghost.state == "hunt":
            break
    assert ghost.state == "hunt", f"eaten ghost never respawned: {ghost}"
    print("eaten-ghost check ok (200 pts, BFS home, respawn)")


def check_death_reset_and_game_over():
    game = Game(decide=StubClient(3).decide, seed=3)
    eaten_before = len(game.pellets)
    game.pellets.discard((1, 1))          # pretend this pellet was eaten earlier
    ghost = game.ghosts[0]
    ghost.state, ghost.fright = "hunt", 0.0
    ghost.x, ghost.y = game.pac.x, game.pac.y
    game.resolve_collisions()
    assert game.lives == START_LIVES - 1, game.lives
    assert (game.pac.x, game.pac.y) == PAC_SPAWN, "positions must reset"
    assert len(game.pellets) == eaten_before - 1, "pellets stay eaten"

    game.lives = 1
    ghost = game.ghosts[0]
    ghost.state, ghost.fright = "hunt", 0.0
    ghost.x, ghost.y = game.pac.x, game.pac.y
    game.resolve_collisions()
    assert game.lives == 0 and game.game_over, "0 lives -> game over"
    game.restart()
    assert game.lives == START_LIVES and game.score == 0 and game.level == 1
    assert not game.game_over and len(game.pellets) + len(game.power) == 244
    print("death/game-over/restart check ok")


def check_chase():
    game = Game(decide=StubClient(0).decide, seed=0)
    ghost = game.ghosts[0]
    ghost.state, ghost.fright = "hunt", 0.0
    ghost.x, ghost.y, ghost.dir = 1, 5, "up"       # the top-left corner loop
    game.pac.x, game.pac.y = 1, 29
    legal = game.legal_moves(ghost.x, ghost.y, ghost.dir)
    assert legal == ["up", "right"], legal
    dist = game.distance_field((1, 29))
    assert dist[(2, 5)] < dist[(1, 4)]              # true path goes right, not up
    assert game.chase_move(ghost, legal) == "right", "chase must follow the shortest path"

    ghost.x, ghost.y, ghost.dir = 26, 14, "up"     # tunnel row, Pac at (1, 14)
    game.pac.x, game.pac.y = 1, 14
    legal = game.legal_moves(ghost.x, ghost.y, ghost.dir)
    assert legal == ["left", "right"], legal
    assert game.chase_move(ghost, legal) == "right", "chase must use the tunnel wrap"
    print("chase check ok (BFS shortest path to Pac, wrap-aware, never reverses)")


def check_state_text_pellets():
    game = Game(decide=StubClient(0).decide, seed=0)
    game.pellets.discard((1, 1))
    row = game.state_text(["up"]).splitlines()[1]
    assert row[1] == " ", "an eaten pellet must not show in Laya's state"
    assert row[2] == ".", "a present pellet must show in Laya's state"
    print("state-text check ok (eaten pellets cleared)")


def check_tunnels():
    game = Game(decide=StubClient(4).decide, seed=4)
    game.pac.x, game.pac.y, game.pac.dir = 1, 14, "left"
    game.step_pac()
    assert (game.pac.x, game.pac.y) == (0, 14), (game.pac.x, game.pac.y)
    game.step_pac()
    assert (game.pac.x, game.pac.y) == (COLS - 1, 14), (game.pac.x, game.pac.y)

    game = Game(decide=StubClient(5).decide, seed=5)
    ghost = game.ghosts[0]
    ghost.state, ghost.dir, ghost.x, ghost.y = "hunt", "left", 1, 14
    game.step_ghost(ghost)
    game.step_ghost(ghost)
    assert (ghost.x, ghost.y) == (COLS - 1, 14), (ghost.x, ghost.y)
    print("tunnel-wrap check ok (pac and ghosts)")


def check_maze_shape():
    maze = load_maze()
    assert len(maze) == ROWS and all(len(r) == COLS for r in maze)
    dots = sum(row.count(".") for row in maze)
    powers = sum(row.count("o") for row in maze)
    assert powers == 4, "4 power pellets"
    assert dots + powers == 244, "classic 244 pellets (240 dots + 4 energizers)"
    print("maze-shape check ok")


def check_laya_presentation():
    """Laya client guards, presentation modes, mapping determinism, cache."""
    from laya_client import (PRESENTATION_MODES, LayaClient, format_features_state,
                             option_mapping)

    class FakeAgent:
        def __init__(self, pick=None, probs_fn=None, usage=None):
            self.calls = []
            self.device = type("Dev", (), {"type": "cuda"})()
            self.pick, self.probs_fn, self.usage = pick, probs_fn, usage or {}

        def predict(self, state, questions):
            self.calls.append((state, questions))
            keys = list(questions["move"]["criteria"])
            if self.probs_fn:
                probs = self.probs_fn(keys)
            elif self.pick:
                probs = {k: (0.9 if k == self.pick else 0.1 / max(1, len(keys) - 1)) for k in keys}
            else:
                probs = {k: 1.0 / len(keys) for k in keys}
            return {"answers": {"move": {"probabilities": probs}}, "usage": dict(self.usage)}

    def raises(fn, frag):
        try:
            fn()
        except RuntimeError as e:
            assert frag in str(e), f"expected {frag!r} in {e}"
            return
        raise AssertionError(f"no RuntimeError containing {frag!r}")

    raises(lambda: LayaClient(agent=FakeAgent(usage={"truncated": True})).decide("s", "i", ("up", "left")), "truncated")
    raises(lambda: LayaClient(agent=FakeAgent(usage={"options": {"up": 0}})).decide("s", "i", ("up", "left")), "collapsed")
    raises(lambda: LayaClient(agent=FakeAgent(probs_fn=lambda k: {"A": 1.0})).decide("s", "i", ("up", "left")), "were presented")
    raises(lambda: LayaClient(agent=FakeAgent(probs_fn=lambda k: {x: 0.9 for x in k})).decide("s", "i", ("up", "left")), "sum to ~1.0")

    class NoAnswer(FakeAgent):
        def predict(self, state, questions):
            return {"answers": {}, "usage": {}}

    raises(lambda: LayaClient(agent=NoAnswer()).decide("s", "i", ("up", "left")), "no 'move' answer")

    for mode in PRESENTATION_MODES:
        for legal in (("up", "left"), ("up", "down", "left")):
            mapping = option_mapping("stateX", mode, legal)
            assert set(mapping.values()) == set(legal)
            for key, direction in mapping.items():
                client = LayaClient(agent=FakeAgent(pick=key), mode=mode)
                assert client.decide("stateX", "i", legal) == direction, (mode, key, mapping)
                assert set(client.last["probs"]) == set(legal)

    assert option_mapping("stateX", "labels", ("up", "left")) == {"up": "up", "left": "left"}
    golden = option_mapping("stateX", "neutral", ("up", "down", "left"))
    assert golden == {"A": "left", "B": "up", "C": "down"}, golden
    assert option_mapping("stateX", "neutral", ("up", "down", "left")) == golden

    agent = FakeAgent()
    client = LayaClient(agent=agent)
    first = client.decide("stateA", "i", ("up", "left"))
    assert len(agent.calls) == 1 and client.last["cached"] is False
    assert client.decide("stateA", "i", ("up", "left")) == first
    assert len(agent.calls) == 1 and client.last["cached"] is True
    LayaClient(agent=agent, mode="neutral").decide("stateA", "i", ("up", "left"))
    assert len(agent.calls) == 2, "mode switch must miss"
    assert LayaClient(agent=FakeAgent()).decide("s", "i", ("left",)) == "left"

    features = {
        "up": {"pellet": 7, "zone": 8, "power": 18, "danger": "low", "edible": None, "new": 0.6, "seen": False},
        "left": {"pellet": 2, "zone": 2, "power": 20, "danger": "low", "edible": None, "new": 0.0, "seen": True},
    }
    meta = {"junction": (6, 5), "visits": 3, "came_from": "right", "mode": "exploring"}
    for mode in PRESENTATION_MODES:
        state, instructions, mapping = format_features_state(features, meta, mode)
        assert "moves:" in state and "#" not in state, state
        if mode == "neutral":
            assert not any(w in state or w in instructions
                           for w in ("up", "down", "left", "right")), (state, instructions)

    # letter/mapping agreement through decide_features: A must map to the same
    # direction the formatter rendered under A
    for mode in ("shuffled", "neutral"):
        client = LayaClient(agent=FakeAgent(pick="A"), mode=mode)
        got = client.decide_features(features, ("up", "left"), meta)
        assert got == client.last["mapping"]["A"], (mode, got, client.last["mapping"])
        assert client.last["position"] == 0

    # unreachable distances render as "-" (energizers eaten -> power is empty)
    inf_state, _, _ = format_features_state(
        {"up": {"pellet": 1 << 30, "zone": 0, "power": 1 << 30, "danger": "low",
                "edible": None, "new": 0.0, "seen": False}}, {}, "labels")
    assert "1073741824" not in inf_state, inf_state
    assert "pellet -" in inf_state and "power -" in inf_state, inf_state
    print("laya-presentation check ok (guards, 3 modes, mapping, cache, features-only)")


def check_brain_modes():
    """WP5: personality mode state machine (exploring/hunting/escaping/chasing/recovering)."""
    from brain import HUNT_PELLETS, RECOVER_TICKS

    def stub(state, instructions, legal):
        return legal[0]

    def setup():
        game = Game(decide=stub, seed=0)
        for gh in game.ghosts:
            gh.state, gh.fright = "house", 0.0
        game.pac.x, game.pac.y, game.pac.dir = 6, 5, "up"
        return game

    def mode_of(game):
        legal = game.legal_moves(game.pac.x, game.pac.y, game.pac.dir)
        game.brain.choose(game, legal)
        return game.brain.last["mode"]

    game = setup()
    assert mode_of(game) == "exploring", game.brain.last

    game = setup()
    game.ghosts[0].state, game.ghosts[0].x, game.ghosts[0].y = "hunt", 6, 3
    assert mode_of(game) == "escaping", game.brain.last

    game = setup()
    game.ghosts[0].state, game.ghosts[0].fright = "hunt", 5.0
    game.ghosts[0].x, game.ghosts[0].y = 5, 5
    assert mode_of(game) == "chasing", game.brain.last

    game = setup()
    game.pellets = set(list(game.pellets)[:HUNT_PELLETS - 1])
    game.power.clear()
    assert mode_of(game) == "hunting", game.brain.last

    game = setup()
    assert mode_of(game) == "exploring"
    game.lives -= 1
    assert mode_of(game) == "recovering", game.brain.last
    for _ in range(RECOVER_TICKS - 1):
        assert mode_of(game) == "recovering", game.brain.last
    assert mode_of(game) == "exploring", game.brain.last
    print("brain-modes check ok (exploring/escaping/chasing/hunting/recovering)")


def check_smooth_render():
    """Renderer interpolates sprites between tiles; tunnel wrap snaps."""
    from game import Renderer

    game = Game(decide=lambda s, i, legal: legal[0], seed=0)
    game.step()                              # Pac moved: prev and tile differ
    ent = game.pac
    assert ent.prev != (ent.x, ent.y), "expected a move to test interpolation"
    assert Renderer._pos(ent, 0.0) == (float(ent.prev[0]), float(ent.prev[1]))
    assert Renderer._pos(ent, 1.0) == (float(ent.x), float(ent.y))
    mx, my = Renderer._pos(ent, 0.5)
    assert abs(mx - (ent.prev[0] + ent.x) / 2) < 1e-9
    assert abs(my - (ent.prev[1] + ent.y) / 2) < 1e-9
    ent.prev, ent.x, ent.y = (0, 14), 27, 14  # tunnel wrap
    assert Renderer._pos(ent, 0.5) == (27.0, 14.0)
    print("smooth-render check ok (interpolation + tunnel snap)")


def run_live():
    from laya_client import LayaClient

    client = LayaClient()
    print(f"live: model loaded on {client.agent.device}, warmup done")

    game = Game(decide=client.decide, seed=0)
    game.pac.x, game.pac.y, game.pac.dir = 6, 5, "up"   # a real 3-way junction
    legal = game.legal_moves(game.pac.x, game.pac.y, game.pac.dir)
    assert len(legal) >= 2, legal
    state, instructions = game.state_text(legal), game.instructions(legal)
    move = client.decide(state, instructions, tuple(legal))
    assert move in legal, move
    last = client.last
    assert not last["truncated"], "model saw a truncated maze"
    assert not last["options"], "question options collapsed"
    assert set(last["probs"]) == set(legal), last["probs"]
    total = sum(last["probs"].values())
    assert abs(total - 1.0) < 0.02, total
    print("live: one real decision ok")
    print("  probabilities: {" + ", ".join(f"{k}: {v:.3f}" for k, v in last["probs"].items()) + "}")
    print(f"  choice: {last['choice']}  latency: {last['ms']:.0f} ms")

    game.step_pac()
    assert (game.pac.x, game.pac.y) != (6, 5), "game did not move after a real decision"
    print("live: game integrated with the model ok")

    # presentation modes on the same model instance (no extra VRAM)
    mv = client.decide(state, instructions, tuple(legal), mode="shuffled")
    assert mv in legal and set(client.last["probs"]) == set(legal), client.last
    assert abs(sum(client.last["probs"].values()) - 1.0) < 0.02
    print(f"live: shuffled mode ok -> {mv}  mapping={client.last['mapping']}")

    features = {
        "up": {"pellet": 7, "zone": 8, "power": 18, "danger": "low", "edible": None, "new": 0.6, "seen": False},
        "left": {"pellet": 2, "zone": 2, "power": 20, "danger": "low", "edible": None, "new": 0.0, "seen": True},
        "right": {"pellet": 3, "zone": 15, "power": 9, "danger": "med", "edible": 4, "new": 0.8, "seen": False},
    }
    meta = {"junction": (6, 5), "visits": 1, "came_from": "right", "mode": "exploring"}
    mv = client.decide_features(features, tuple(legal), meta, mode="neutral")
    assert mv in legal and set(client.last["probs"]) == set(legal), client.last
    assert not client.last["truncated"] and not client.last["options"]
    assert client.last["state_tokens"] is None or client.last["state_tokens"] < 397, client.last
    print(f"live: features-only neutral mode ok -> {mv}  state_tokens={client.last['state_tokens']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="GPU smoke test: real Laya model on CUDA")
    args = parser.parse_args(argv)

    check_maze_shape()
    check_level_clears()
    check_power_pellet()
    check_eat_ghost_and_eyes_home()
    check_death_reset_and_game_over()
    check_chase()
    check_state_text_pellets()
    check_laya_presentation()
    check_brain_modes()
    check_smooth_render()
    check_tunnels()
    check_simulations()
    print("all stub checks passed")
    if args.live:
        run_live()
        print("live checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
