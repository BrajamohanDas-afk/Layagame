"""Anti-bias experiment (WP3): labels vs shuffled vs neutral presentation.

  uv run python bias_experiment.py --collect 400                  # frozen corpus, no GPU
  uv run python bias_experiment.py --replay --fake --limit 400    # harness negative control
  uv run python bias_experiment.py --replay --live --limit 400    # real Laya, all 3 modes

Measures whether a direction preference is (a) directional semantic bias,
(b) option-position bias, or (c) genuine feature preference, using
availability-matched nulls (every decision can only pick from its own legal set).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter

from bench import UniformStub
from game import Game, REVERSE
from laya_client import LayaClient, format_features_state

DIRECTIONS = ("up", "down", "left", "right")
CORPUS = "bias_corpus.jsonl"


class FakeAgent:
    """Jittered fake model for the harness negative control (no exact ties)."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.device = type("Dev", (), {"type": "cuda"})()

    def predict(self, state, questions):
        keys = list(questions["move"]["criteria"])
        vals = {k: self.rng.random() + 0.01 for k in keys}
        total = sum(vals.values())
        return {"answers": {"move": {"probabilities": {k: v / total for k, v in vals.items()}}},
                "usage": {}}


def collect(count, seed=0, max_steps=20000):
    game = Game(decide=UniformStub(seed).decide, seed=seed)
    entries = []
    original = game.brain.choose

    def spy(g, legal):
        features = g.brain.features(g, legal)
        junction = (g.pac.x, g.pac.y)
        entry = {
            "legal": list(legal),
            "features": features,
            "board": g.state_text(list(legal)),
            "board_instructions": g.instructions(list(legal)),
            "meta": {"junction": list(junction),
                     "visits": g.brain.mem.visit_count(junction),
                     "came_from": REVERSE[g.pac.dir],
                     "mode": g.brain.mode},
        }
        entries.append(entry)
        move, source = original(g, legal)
        entry["meta"]["mode"] = g.brain.mode   # the mode this decision actually used
        return move, source

    game.brain.choose = spy
    for _ in range(max_steps):
        if game.game_over:
            game.restart()
            game.brain.choose = spy
        game.step()
        if len(entries) >= count:
            break
    with open(CORPUS, "w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    print(f"collected {len(entries)} junction states -> {CORPUS}")
    return entries


def load_corpus(limit=None):
    entries = []
    with open(CORPUS, encoding="utf-8") as fh:
        for line in fh:
            entries.append(json.loads(line))
            if limit and len(entries) >= limit:
                break
    return entries


def replay(entries, client, mode, label):
    rows = []
    for e in entries:
        legal = tuple(e["legal"])
        state, instructions, mapping = format_features_state(
            e["features"], e["meta"], mode)
        move = client.decide(state, instructions, legal, mode=mode, mapping=mapping)
        last = client.last
        rows.append({"legal": legal, "chosen": move, "mapping": dict(mapping),
                     "position": last["position"], "probs": dict(last["probs"]),
                     "cached": last["cached"],
                     "_pellet": {m: e["features"][m].get("pellet", 10 ** 6)
                                 for m in e["features"]}})
    print(f"replayed {len(rows)} decisions in {label} mode")
    return rows


def analyze(rows, mode):
    n = len(rows)
    k_dist = Counter(len(r["legal"]) for r in rows)
    dir_obs = Counter(r["chosen"] for r in rows)
    dir_exp = Counter()
    dir_var = Counter()
    pos_obs = Counter(r["position"] for r in rows)
    pos_exp = Counter()
    for r in rows:
        k = len(r["legal"])
        for d in r["legal"]:
            dir_exp[d] += 1.0 / k
            dir_var[d] += (1.0 / k) * (1.0 - 1.0 / k)
        for p in range(k):
            pos_exp[p] += 1.0 / k

    def chi2(obs, exp):
        total = 0.0
        for key in sorted(exp):
            e = exp[key]
            if e > 0:
                total += (obs.get(key, 0) - e) ** 2 / e
        return total

    z = {}
    for d in DIRECTIONS:
        var = dir_var.get(d, 0.0)
        if var > 0:
            z[d] = (dir_obs.get(d, 0) - dir_exp[d]) / math.sqrt(var)

    lr = [r for r in rows if set(r["legal"]) == {"left", "right"}]
    lr_left = sum(1 for r in lr if r["chosen"] == "left")

    agree = 0
    chance = 0.0
    counted = 0
    ties = 0
    for r in rows:
        pellets = r["_pellet"]
        best_val = min(pellets.get(m, 10 ** 6) for m in r["legal"])
        bests = [m for m in r["legal"] if pellets.get(m, 10 ** 6) == best_val]
        if len(bests) == 1:                     # skip tied bests: credit is ambiguous
            counted += 1
            chance += 1.0 / len(r["legal"])
            agree += 1 if r["chosen"] == bests[0] else 0
        top = sorted(r["probs"].values(), reverse=True)
        if len(top) >= 2 and top[0] - top[1] < 0.02:
            ties += 1

    return {
        "mode": mode, "n": n, "k_dist": dict(k_dist),
        "direction_obs": {d: dir_obs.get(d, 0) for d in DIRECTIONS},
        "direction_expected": {d: round(dir_exp.get(d, 0.0), 1) for d in DIRECTIONS},
        "direction_z": {d: round(v, 2) for d, v in z.items()},
        "chi2_direction": round(chi2(dir_obs, dir_exp), 1),
        "position_obs": dict(pos_obs),
        "position_expected": {p: round(v, 1) for p, v in pos_exp.items()},
        "chi2_position": round(chi2(pos_obs, pos_exp), 1),
        "left_right_n": len(lr), "left_right_left": lr_left,
        "feature_agreement": round(agree / max(1, counted), 3),
        "chance_agreement": round(chance / max(1, counted), 3),
        "agreement_rows": counted,
        "near_ties": ties,
    }


def ab_features_vs_board(entries, client, limit):
    """Replay the same junctions through the board prompt and the features prompt."""
    rows = {"board": [], "features": []}
    for e in entries[:limit]:
        legal = tuple(e["legal"])
        pellets = {m: e["features"][m].get("pellet", 10 ** 6) for m in e["features"]}

        move = client.decide(e["board"], e["board_instructions"], legal)
        last = dict(client.last) if client.last else {}
        rows["board"].append({"legal": legal, "chosen": move,
                              "tokens": last.get("state_tokens"), "ms": last.get("ms"),
                              "cached": bool(last.get("cached")),
                              "truncated": bool(last.get("truncated")),
                              "_pellet": pellets})

        move = client.decide_features(e["features"], legal, e["meta"])
        last = dict(client.last) if client.last else {}
        rows["features"].append({"legal": legal, "chosen": move,
                                 "tokens": last.get("state_tokens"), "ms": last.get("ms"),
                                 "cached": bool(last.get("cached")),
                                 "truncated": bool(last.get("truncated")),
                                 "_pellet": pellets})
    return rows


def ab_summary(rows):
    n = len(rows)
    tokens = [r["tokens"] for r in rows if r["tokens"] is not None]
    ms = [r["ms"] for r in rows if r["ms"] is not None]
    agree = counted = 0
    chance = 0.0
    dirs = Counter()
    for r in rows:
        dirs[r["chosen"]] += 1
        best_val = min(r["_pellet"].get(m, 10 ** 6) for m in r["legal"])
        bests = [m for m in r["legal"] if r["_pellet"].get(m, 10 ** 6) == best_val]
        if len(bests) == 1:
            counted += 1
            chance += 1.0 / len(r["legal"])
            agree += 1 if r["chosen"] == bests[0] else 0
    return {
        "n": n,
        "mean_tokens": round(sum(tokens) / len(tokens), 1) if tokens else None,
        "max_tokens": max(tokens) if tokens else None,
        "mean_ms": round(sum(ms) / len(ms), 1) if ms else None,
        "cached": sum(1 for r in rows if r["cached"]),
        "truncated": sum(1 for r in rows if r["truncated"]),
        "directions": dict(dirs),
        "feature_agreement": round(agree / max(1, counted), 3),
        "chance_agreement": round(chance / max(1, counted), 3),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collect", type=int, metavar="N")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--ab", action="store_true",
                    help="features-only vs board prompt A/B on the frozen corpus")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.collect:
        collect(args.collect)
        return 0

    if args.ab:
        entries = load_corpus(args.limit)
        if not entries:
            ap.error(f"{CORPUS} is empty; run --collect first")
        client = LayaClient(agent=FakeAgent()) if args.fake else LayaClient()
        rows = ab_features_vs_board(entries, client, args.limit)
        summary = {k: ab_summary(v) for k, v in rows.items()}
        out = args.out or "ab_features.json"
        print(f"\n{'path':10} {'n':>5} {'mean_tok':>9} {'max_tok':>8} "
              f"{'mean_ms':>8} {'trunc':>6} {'feature':>8} {'chance':>7}")
        for path, s in summary.items():
            print(f"{path:10} {s['n']:>5} {str(s['mean_tokens']):>9} "
                  f"{str(s['max_tokens']):>8} {str(s['mean_ms']):>8} "
                  f"{s['truncated']:>6} {s['feature_agreement']:>8} {s['chance_agreement']:>7}")
        for path, s in summary.items():
            print(f"[{path}] directions={s['directions']}")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"live": not args.fake, "limit": args.limit,
                       "summary": summary}, fh, indent=1, sort_keys=True)
        print(f"\nwrote {out}")
        return 0

    if not args.replay:
        ap.error("pass --collect N, --replay or --ab")
    entries = load_corpus(args.limit)
    if not entries:
        ap.error(f"{CORPUS} is empty; run --collect first")

    client = LayaClient(agent=FakeAgent()) if args.fake else LayaClient()
    results = []
    for mode in ("labels", "shuffled", "neutral"):
        rows = replay(entries, client, mode, mode)
        results.append(analyze(rows, mode))

    print(f"\n{'mode':10} {'n':>5} {'chi2_dir':>9} {'chi2_pos':>9} "
          f"{'LR-left':>8} {'feature':>8} {'chance':>7}")
    for r in results:
        lr = f"{r['left_right_left']}/{r['left_right_n']}" if r["left_right_n"] else "-"
        print(f"{r['mode']:10} {r['n']:>5} {r['chi2_direction']:>9} "
              f"{r['chi2_position']:>9} {lr:>8} "
              f"{r['feature_agreement']:>8} {r['chance_agreement']:>7}")
    for r in results:
        print(f"\n[{r['mode']}] directions obs={r['direction_obs']} "
              f"exp={r['direction_expected']} z={r['direction_z']}")
        print(f"[{r['mode']}] positions obs={r['position_obs']} exp={r['position_expected']}")
    out = args.out or "bias_results.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"live": not args.fake, "limit": args.limit,
                   "results": results}, fh, indent=1, sort_keys=True)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
