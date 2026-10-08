"""Benchmark harness for the Laya-Pacman decision architecture (WP0/WP4).

Collects the metrics from DECISION-ARCHITECTURE.html §09 in headless runs:

  uv run python bench.py --steps 1500 --seeds 0 1 2                  # stub, no GPU
  uv run python bench.py --steps 400 --seeds 0 --live --mode labels  # real model
  uv run python bench.py --out bench_baseline.json

One JSON row per (config, seed) plus an aggregate.  Stub runs are fully
deterministic; live runs report per-call latency and token usage.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, deque
from statistics import mean

from game import COLS, Game
from laya_client import LayaClient, canonical_legal

STUCK_AFTER = 30          # steps without a pellet -> one stuck event
REVISIT_WINDOW = 50       # junction memory for the revisit metric


class UniformStub:
    """Deterministic stand-in for Laya that exposes the same .last contract."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.last = None
        self.calls = 0
        self.cache_hits = 0
        self.guard_failures = Counter()

    def decide(self, state, instructions, legal):
        legal = canonical_legal(legal)
        move = self.rng.choice(legal)
        self.calls += 1
        self.last = {
            "legal": legal, "probs": {m: 1.0 / len(legal) for m in legal},
            "choice": move, "position": legal.index(move),
            "mapping": {d: d for d in legal}, "mode": "labels", "ms": 0.0,
            "cached": False, "truncated": None, "options": None,
            "state_tokens": max(1, len(state) // 4),
        }
        return move


class LiveCollector:
    """Wrap LayaClient; expose .decide/.decide_features and count guard failures."""

    def __init__(self, client):
        self.client = client
        self.last = None
        self.calls = 0
        self.cache_hits = 0
        self.guard_failures = Counter()
        self.truncated = 0

    def _call(self, fn):
        try:
            move = fn()
        except RuntimeError as e:
            msg = str(e)
            if "truncated" in msg:
                self.truncated += 1
                self.guard_failures["truncated"] += 1
            elif "collapsed" in msg:
                self.guard_failures["options"] += 1
            elif "sum to" in msg or "non-finite" in msg:
                self.guard_failures["sum"] += 1
            elif "were presented" in msg:
                self.guard_failures["options_returned"] += 1
            else:
                self.guard_failures["other"] += 1
            raise
        self.last = dict(self.client.last) if self.client.last else None
        if self.last and self.last["cached"]:
            self.cache_hits += 1
        else:
            self.calls += 1
        return move

    def decide(self, state, instructions, legal):
        return self._call(lambda: self.client.decide(state, instructions, legal))

    def decide_features(self, features, legal, meta=None):
        return self._call(lambda: self.client.decide_features(features, legal, meta=meta))


def run(seed, steps, client, ablate=False):
    game = Game(decide=client.decide, seed=seed)
    m = {
        "steps": 0, "decisions": 0, "pellets_eaten": 0, "deaths": 0,
        "game_overs": 0, "restarts": 0, "score": 0, "levels": 0,
        "stuck_events": 0, "revisits": 0, "truncated": 0, "stuck_remaining": [],
        "sources": Counter(), "directions": Counter(), "laya_directions": Counter(),
        "positions": Counter(), "laya_positions": Counter(),
        "tokens": [], "ms": [], "laya_decisions": 0, "cached": 0,
    }
    recent_junctions = deque(maxlen=REVISIT_WINDOW)
    no_pellet = 0
    stuck_armed = True
    decisions = []
    d_seen = 0

    original_choose = game.brain.choose

    if ablate:
        # Pre-architecture baseline: board prompt + raw model choice, no memory,
        # no controllers, no aggregator.
        def spy(g, legal):
            move = client.decide(g.state_text(list(legal)),
                                 g.instructions(list(legal)), tuple(legal))
            decisions.append({"junction": (g.pac.x, g.pac.y), "move": move,
                              "source": "laya"})
            return move, "laya"
    else:
        def spy(g, legal):
            move, source = original_choose(g, legal)
            decisions.append({"junction": g.brain.last["junction"], "move": move,
                              "source": source})
            return move, source

    game.brain.choose = spy

    for _ in range(steps):
        if game.game_over:
            m["game_overs"] += 1
            game.restart()
            game.brain.choose = spy
            recent_junctions.clear()
            m["restarts"] += 1
        before_pellets = len(game.pellets) + len(game.power)
        lives_before = game.lives
        game.step()
        m["steps"] += 1
        after_pellets = len(game.pellets) + len(game.power)
        eaten = before_pellets if after_pellets > before_pellets else before_pellets - after_pellets
        m["pellets_eaten"] += max(0, eaten)
        if game.lives < lives_before:
            m["deaths"] += lives_before - game.lives
        m["score"] = max(m["score"], game.score)
        m["levels"] = max(m["levels"], game.level)

        if eaten:
            no_pellet = 0
            stuck_armed = True
        else:
            no_pellet += 1
            if no_pellet >= STUCK_AFTER and stuck_armed:
                m["stuck_events"] += 1
                m["stuck_remaining"].append(len(game.pellets) + len(game.power))
                stuck_armed = False

        while len(decisions) > d_seen:
            entry = decisions[d_seen]
            d_seen += 1
            m["decisions"] += 1
            m["sources"][entry["source"]] += 1
            m["directions"][entry["move"]] += 1
            junction = entry["junction"]
            if junction in recent_junctions:
                m["revisits"] += 1
            recent_junctions.append(junction)
            if entry["source"] == "laya" and client.last is not None:
                m["laya_decisions"] += 1
                last = client.last
                m["laya_directions"][last["choice"]] += 1
                m["laya_positions"][last["position"]] += 1
                m["cached"] += 1 if last["cached"] else 0
                m["tokens"].append(last.get("state_tokens"))
                m["ms"].append(last.get("ms") or 0.0)

    m["truncated"] = getattr(client, "truncated", 0)
    m["client"] = client
    return m


def summarize(m):
    steps = max(1, m["steps"])
    dec = max(1, m["decisions"])
    laya = max(1, m["laya_decisions"])
    tokens = [t for t in m["tokens"] if t is not None]
    return {
        "steps": m["steps"],
        "decisions": m["decisions"],
        "pellets_per_1000": round(1000.0 * m["pellets_eaten"] / steps, 1),
        "deaths_per_1000": round(1000.0 * m["deaths"] / steps, 2),
        "stuck_events_per_1000": round(1000.0 * m["stuck_events"] / steps, 2),
        "stuck_events_midlevel": sum(1 for r in m["stuck_remaining"] if r > 100),
        "stuck_events_endgame": sum(1 for r in m["stuck_remaining"] if r <= 100),
        "revisits_per_1000": round(1000.0 * m["revisits"] / steps, 2),
        "revisit_share": round(m["revisits"] / dec, 3),
        "sources": dict(m["sources"]),
        "laya_share": round(m["sources"].get("laya", 0) / dec, 3),
        "direction_freq": dict(m["directions"]),
        "laya_direction_freq": dict(m["laya_directions"]),
        "option_position_freq": dict(m["laya_positions"]),
        "mean_state_tokens": round(mean(tokens), 1) if tokens else None,
        "max_state_tokens": max(tokens) if tokens else None,
        "mean_ms": round(mean(m["ms"]), 1) if m["ms"] else None,
        "cache_hit_rate": round(m["cached"] / laya, 3),
        "truncated": m["truncated"],
        "guard_failures": dict(getattr(m.get("client"), "guard_failures", {})),
        "score": m["score"],
        "levels": m["levels"],
        "game_overs": m["game_overs"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--live", action="store_true", help="use the real Laya model on CUDA")
    ap.add_argument("--mode", default="labels", choices=("labels", "shuffled", "neutral"))
    ap.add_argument("--ablate", action="store_true",
                    help="pre-architecture baseline: board prompt + raw model choice")
    ap.add_argument("--out", default=None, help="write the full report as JSON")
    args = ap.parse_args(argv)

    if args.live:
        client = LiveCollector(LayaClient(mode=args.mode))
    else:
        client = UniformStub(seed=args.seeds[0])

    rows = []
    for seed in args.seeds:
        if isinstance(client, UniformStub):
            client = UniformStub(seed=seed)
        elif isinstance(client, LiveCollector):
            client.client.cache.clear()   # keep seeds independent
        m = run(seed, args.steps, client, ablate=args.ablate)
        s = summarize(m)
        s["seed"] = seed
        rows.append(s)
        print(f"seed {seed}: pellets/1k={s['pellets_per_1000']} deaths/1k={s['deaths_per_1000']} "
              f"revisits/1k={s['revisits_per_1000']} stuck/1k={s['stuck_events_per_1000']} "
              f"laya_share={s['laya_share']} tokens={s['mean_state_tokens']}")

    keys = ("pellets_per_1000", "deaths_per_1000", "stuck_events_per_1000",
            "revisits_per_1000", "revisit_share", "laya_share", "cache_hit_rate")
    agg = {k: round(mean(r[k] for r in rows), 3) for k in keys}
    sources = Counter()
    for r in rows:
        sources.update(r["sources"])
    report = {
        "config": {"steps": args.steps, "seeds": args.seeds, "live": args.live,
                   "mode": args.mode, "ablate": args.ablate},
        "aggregate": {**agg, "sources": dict(sources)},
        "rows": rows,
    }
    print("aggregate:", json.dumps(report["aggregate"]))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
