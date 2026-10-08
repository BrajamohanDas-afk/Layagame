"""Engine brain for Laya-Pacman: all spatial reasoning, memory, loop detection,
deterministic controllers, and the final aggregator.

Laya never does pathfinding, geometry, distance or traversal reasoning.  The
engine computes facts (features) per legal move at a junction; deterministic
controllers answer situations with a known right answer; Laya only makes the
strategic choice among the remaining reasonable options.

Pipeline (see docs/DECISION-ARCHITECTURE.html):

    game.step_pac() -> Brain.choose(game, legal) -> (move, source)
        danger  -> deterministic safety
        loop    -> deterministic escape / exploration
        chase   -> deterministic chase of an edible ghost
        laya    -> aggregator: model probs + engine feature scores
"""

from __future__ import annotations

from collections import deque

DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
REVERSE = {"up": "down", "down": "up", "left": "right", "right": "left"}
DIR_ORDER = ("up", "down", "left", "right")

INF = 1 << 30

# ---- feature extraction -------------------------------------------------
ZONE_RADIUS = 12        # BFS radius for the zone count and the exploration score
DANGER_CLOSE = 2        # "within 2 tiles and closing" -> high / safety controller
DANGER_NEAR = 4         # med when a deadly ghost is this close
EDIBLE_RADIUS = 8       # chase only when the edible ghost is this close
STAGNATION_N = 20       # no-pellet moves before the fallback escape fires

# ---- personality modes (WP5) ---------------------------------------------
MODES = ("escaping", "chasing", "recovering", "hunting", "exploring")  # priority order
ESCAPE_RADIUS = 3       # deadly ghost this close and closing -> escaping
RECOVER_TICKS = 12      # decisions spent recovering after a death
HUNT_PELLETS = 40       # endgame: collect the last pellets efficiently

# ---- memory & loop detection --------------------------------------------
WINDOW = 16             # rolling junction decisions kept
CYCLE_PERIODS = (2, 3, 4)
CYCLE_REPEATS = 2       # full repeats required to fire a cycle
REPEATS_BEFORE_FIRE = 2  # prior identical (tile, entered-from) states
DECAY_EVERY = 10        # decisions between visit-count decay passes
DECAY_NUM, DECAY_DEN = 7, 8   # half-life ~52 decisions
FORBID_TICKS = 24       # how long a loop-closing branch stays forbidden

# ---- aggregator weights (WP4-tuned 2026-10-06 on live 5x1000 A/B) --------
W_MODEL = 1.0
W_PELLET = 0.8
W_EXPLORE = 0.5
W_LOOP = 0.5
W_DANGER = 0.8

DANGER_PENALTY = {"high": 1.0, "med": 0.5, "low": 0.0}
SOURCES = ("danger", "loop", "chase", "laya", "forced")

INSTRUCTIONS = ("Pick the best move for Pac-Man: stay alive first, "
                "then eat pellets, then explore new ground.")


def _bfs_field(game, sources):
    """Multi-source BFS step distance over Pac-passable tiles (door blocked).

    Returns {tile: steps}; tiles absent from the map are unreachable.
    """
    dist = {}
    queue = deque()
    for tile in sources:
        dist[tile] = 0
        queue.append(tile)
    while queue:
        x, y = queue.popleft()
        step = dist[(x, y)] + 1
        for m in DIR_ORDER:
            nxt = game.ahead(x, y, m)
            if nxt in dist or not game.passable(*nxt):
                continue
            dist[nxt] = step
            queue.append(nxt)
    return dist


def _dist(value):
    return INF if value is None else value


def _fmt(value):
    return "-" if value >= INF else str(value)


def _fmt_new(value):
    text = f"{value:.1f}"
    return text[1:] if text.startswith("0.") else text


def _danger_level(d, closing):
    """low / med / high from the distance and whether the move closes in."""
    if d <= 2:
        return "high"
    if d <= DANGER_NEAR:
        return "med" if closing else "low"
    return "low"


class Entry:
    __slots__ = ("junction", "entered_from", "chosen", "tick")

    def __init__(self, junction, entered_from, chosen, tick):
        self.junction = junction
        self.entered_from = entered_from
        self.chosen = chosen
        self.tick = tick

    def __repr__(self):
        return (f"<Entry {self.junction} from {self.entered_from} "
                f"chose {self.chosen} @{self.tick}>")


class Memory:
    """Rolling junction memory, visit counts with decay, and loop detection."""

    def __init__(self):
        self.entries = deque(maxlen=WINDOW)
        self.counts = {}
        self.tick = 0
        self.forbidden = {}          # (junction, dir) -> expires_at_tick
        self._level = self._lives = self._remaining = None

    # ------------------------------------------------------------- resets
    def reset_all(self):
        self.entries.clear()
        self.counts.clear()
        self.forbidden.clear()
        self.tick = 0
        self._level = self._lives = self._remaining = None

    def _on_pellet(self):
        self.entries.clear()
        self.forbidden.clear()

    def _on_death_or_level(self):
        self._on_pellet()
        self.counts.clear()

    def _sync(self, game):
        """Detect pellet progress / death / level change from game state."""
        if game is None:
            return
        remaining = len(game.pellets) + len(game.power)
        if self._level is not None:
            if game.level != self._level or game.lives != self._lives:
                self._on_death_or_level()
            elif remaining < self._remaining:
                self._on_pellet()
        self._level, self._lives, self._remaining = game.level, game.lives, remaining

    # ------------------------------------------------------------- writes
    def record(self, game, junction, entered_from, chosen):
        self._sync(game)
        self.tick += 1
        self.entries.append(Entry(junction, entered_from, chosen, self.tick))
        if self.tick % DECAY_EVERY == 0:
            for j in list(self.counts):
                c = self.counts[j] * DECAY_NUM // DECAY_DEN
                if c:
                    self.counts[j] = c
                else:
                    del self.counts[j]
        self.counts[junction] = self.counts.get(junction, 0) + 1
        self.forbidden = {k: t for k, t in self.forbidden.items() if t > self.tick}

    # ------------------------------------------------------------- reads
    def visit_count(self, junction):
        return self.counts.get(junction, 0)

    def seen(self, junction, move):
        return any(e.junction == junction and e.chosen == move for e in self.entries)

    def is_forbidden(self, junction, move):
        return self.forbidden.get((junction, move), 0) > self.tick

    def summary(self, junction, entered_from):
        x, y = junction
        return (f"junction J({x},{y}) visited {max(1, self.visit_count(junction))}x, "
                f"came from {entered_from}")

    # ------------------------------------------------------------- loop detection
    def loop_info(self, game, junction, entered_from):
        """None when no loop; else {kind, period, forbid: (junction, move)|None}."""
        self._sync(game)
        seq = [e.junction for e in self.entries] + [junction]
        for p in CYCLE_PERIODS:
            if len(seq) < CYCLE_REPEATS * p:
                continue
            tail = seq[-CYCLE_REPEATS * p:]
            if (all(tail[i] == tail[i - p] for i in range(p, CYCLE_REPEATS * p))
                    and len(set(tail)) >= p):
                return {"kind": "cycle", "period": p,
                        "forbid": (junction, self.entries[-p].chosen)}
        prior = [e for e in self.entries
                 if e.junction == junction and e.entered_from == entered_from]
        if len(prior) >= REPEATS_BEFORE_FIRE:
            return {"kind": "repeat", "period": None,
                    "forbid": (junction, prior[-1].chosen)}
        if game is not None and getattr(game, "steps_since_pellet", 0) >= STAGNATION_N:
            return {"kind": "stagnation", "period": None, "forbid": None}
        return None

    def escape_move(self, game, junction, entered_from, legal, info):
        """Deterministic escape: forbid the closing branch, allow reversal,
        prefer the least-visited / most-unexplored exit."""
        if info.get("forbid"):
            self.forbidden[info["forbid"]] = self.tick + FORBID_TICKS

        reverse = entered_from          # the true U-turn: back where Pac came from
        reversible = game.passable(*game.ahead(*junction, reverse))
        cands = list(legal)
        if reversible:
            cands.append(reverse)
        cands = [d for d in cands if not self.is_forbidden(junction, d)]

        if not cands:                      # everything forbidden -> unfreeze
            for d in list(legal) + [reverse]:
                self.forbidden.pop((junction, d), None)
            cands = list(legal) + ([reverse] if reversible else [])

        if info["kind"] == "stagnation":   # fallback: head for food if possible
            dist = game.pellet_distance()
            d0 = dist.get(junction, INF)
            closer = [d for d in cands
                      if dist.get(game.ahead(*junction, d), INF) < d0]
            if closer:
                cands = closer

        visited = getattr(game, "visited_since_pellet", None) or ()

        def key(d):
            dest = game.ahead(*junction, d)
            recency = -1
            for i, e in enumerate(self.entries):
                if e.junction == dest:
                    recency = i
            fresh = 0 if dest in visited else 1     # prefer unexplored ground
            return (self.visit_count(dest), -fresh, recency, DIR_ORDER.index(d))

        return min(cands, key=key)


class Brain:
    """choose(game, legal) -> (move, source); owns features, memory, controllers."""

    def __init__(self, decide=None):
        self.decide = decide
        self.mem = Memory()
        self.last = None
        self.history = deque(maxlen=1000)
        self.mode = "exploring"
        self._recover_until = 0
        self._lives_seen = None

    def reset(self):
        self.mem.reset_all()
        self.history.clear()
        self.mode = "exploring"
        self._recover_until = 0
        self._lives_seen = None

    # ------------------------------------------------------------- facts
    def _facts(self, game, legal):
        """Every spatial quantity the decision needs, computed once."""
        here = (game.pac.x, game.pac.y)
        food = game.pellet_distance()
        power = _bfs_field(game, game.power)
        deadly_tiles = [(g.x, g.y) for g in game.ghosts
                        if g.fright <= 0 and g.state not in ("eyes", "house")]
        edible_tiles = [(g.x, g.y) for g in game.ghosts
                        if g.fright > 0 and g.state not in ("eyes", "house")]
        deadly = _bfs_field(game, deadly_tiles)
        deadly_next_tiles = [game.ahead(g.x, g.y, g.dir) for g in game.ghosts
                             if g.fright <= 0 and g.state not in ("eyes", "house")
                             and getattr(g, "dir", None) in DIRS]
        deadly_next = _bfs_field(game, deadly_next_tiles) if deadly_next_tiles else {}
        edible = _bfs_field(game, edible_tiles) if edible_tiles else {}
        d_pac = deadly.get(here, INF)
        d_pac_next = deadly_next.get(here, INF) if deadly_next else INF
        d_pac_edible = edible.get(here, INF) if edible_tiles else INF

        features, closing = {}, {}
        for m in legal:
            dest = game.ahead(*here, m)
            zone, new = self._zone_new(game, dest)
            dd = deadly.get(dest, INF)
            closing[m] = dd < d_pac
            ed = edible.get(dest, INF)
            features[m] = {
                "pellet": _dist(food.get(dest)),
                "zone": zone,
                "power": _dist(power.get(dest)),
                "danger": _danger_level(dd, closing[m]),
                "edible": (None if not edible_tiles else _dist(ed)),
                "new": new,
                "seen": self.mem.seen(here, m),
            }
        return {"here": here, "deadly": deadly, "edible": edible,
                "deadly_next": deadly_next, "d_pac_next": d_pac_next,
                "d_pac": d_pac, "d_pac_edible": d_pac_edible,
                "features": features, "closing": closing}

    def _zone_new(self, game, start):
        """(pellets within radius R, fraction of reachable tiles not visited)."""
        depth = {start: 0}
        queue = deque([start])
        reachable = [start]
        zone = 1 if start in game.pellets else 0
        while queue:
            cur = queue.popleft()
            d = depth[cur]
            if d >= ZONE_RADIUS:
                continue
            for m in DIR_ORDER:
                nxt = game.ahead(*cur, m)
                if nxt in depth or not game.passable(*nxt):
                    continue
                depth[nxt] = d + 1
                queue.append(nxt)
                reachable.append(nxt)
                if nxt in game.pellets:
                    zone += 1
        visited = getattr(game, "visited_since_pellet", None) or ()
        if len(reachable) <= 1:
            return zone, 0.0
        fresh = sum(1 for t in reachable if t not in visited)
        return zone, fresh / len(reachable)

    def features(self, game, legal):
        """Public feature contract: {direction: {pellet, zone, power, danger, edible, new, seen}}."""
        self.mem._sync(game)
        return self._facts(game, legal)["features"]

    # ------------------------------------------------------------- controllers
    def corridor_move(self, game, legal):
        """Forced corridor step: U-turn away from a deadly ghost ahead.

        Forced moves never reach the junction pipeline, so without this a
        deadly ghost in the same corridor would be walked into head-on.  The
        reversal is allowed for safety (step_pac's guard permits it)."""
        here = (game.pac.x, game.pac.y)
        forward = legal[0]
        reverse = REVERSE[game.pac.dir]
        if not game.passable(*game.ahead(*here, reverse)):
            return forward, "forced"
        deadly = [g for g in game.ghosts
                  if g.fright <= 0 and g.state not in ("eyes", "house")]
        if not deadly:
            return forward, "forced"
        field = _bfs_field(game, [(g.x, g.y) for g in deadly])
        next_field = _bfs_field(game, [game.ahead(g.x, g.y, g.dir) for g in deadly
                                       if getattr(g, "dir", None) in DIRS])
        d_pac = field.get(here, INF)
        d_next = next_field.get(here, INF) if next_field else INF
        # Act only on real contact: adjacent now, or a ghost that closes to
        # within 2 after its own move.  A ghost drifting away is left alone.
        if not (d_pac <= 1 or (d_next < d_pac and min(d_pac, d_next) <= 2)):
            return forward, "forced"

        def clearance(m):
            dest = game.ahead(*here, m)
            return min(field.get(dest, INF), next_field.get(dest, INF))

        if clearance(reverse) > clearance(forward):
            self.history.append({"tick": self.mem.tick, "junction": here,
                                 "move": reverse, "source": "danger", "loop": None})
            return reverse, "danger"
        return forward, "forced"

    def _options(self, game, legal):
        """Legal moves plus the reverse when passable (controllers may U-turn)."""
        cands = list(legal)
        reverse = REVERSE[game.pac.dir]
        if reverse not in cands and game.passable(*game.ahead(game.pac.x, game.pac.y, reverse)):
            cands.append(reverse)
        return cands

    def _danger_fires(self, facts):
        """A deadly ghost within 2 tiles fires always; so does one that will be
        within 2 after its own next move (side/behind approaches); within
        DANGER_CLOSE+1 when a legal move closes the gap."""
        d = facts["d_pac"]
        if d <= 2 or facts["d_pac_next"] <= 2:
            return True
        return d <= DANGER_CLOSE + 1 and any(facts["closing"].values())

    def _danger_move(self, game, facts):
        here = game.pac.x, game.pac.y
        reverse = REVERSE[game.pac.dir]

        def key(m):
            dest = game.ahead(*here, m)
            now = facts["deadly"].get(dest, INF)
            after = facts["deadly_next"].get(dest, INF)   # ghost moved one step
            forward = 0 if m != reverse else -1
            return (min(now, after), now, forward, -DIR_ORDER.index(m))

        # If every option is a ghost tile (Pac boxed in) clearance ties at 0 and
        # the first option wins; there is no safe move in that corner.
        return max(self._options(game, list(facts["features"])), key=key)

    def _chase_move(self, game, facts, legal):
        """Move that strictly decreases distance to the nearest edible ghost."""
        if not facts["edible"] or facts["d_pac_edible"] > EDIBLE_RADIUS or facts["d_pac"] <= 1:
            return None
        here = game.pac.x, game.pac.y
        best, best_d = None, facts["d_pac_edible"]
        for m in legal:
            dest = game.ahead(*here, m)
            if facts["deadly"].get(dest, INF) <= 1:   # never chase through a deadly ghost
                continue
            d = facts["edible"].get(dest, INF)
            if d < best_d:
                best, best_d = m, d
        return best

    # ------------------------------------------------------------- model + aggregator
    def _update_mode(self, game, facts):
        """WP5 personality mode for the prompt line.  Deterministic, no extra memory."""
        if self._lives_seen is not None and game.lives != self._lives_seen:
            self._recover_until = self.mem.tick + RECOVER_TICKS   # death or restart
        self._lives_seen = game.lives
        remaining = len(game.pellets) + len(game.power)
        if facts["d_pac"] <= 1 or (facts["d_pac"] <= ESCAPE_RADIUS
                                   and any(facts["closing"].values())):
            self.mode = "escaping"
        elif (facts["edible"] and facts["d_pac_edible"] <= EDIBLE_RADIUS
              and facts["d_pac"] > 1):
            self.mode = "chasing"
        elif self.mem.tick < self._recover_until:
            self.mode = "recovering"
        elif remaining <= HUNT_PELLETS:
            self.mode = "hunting"
        else:
            self.mode = "exploring"
        assert self.mode in MODES
        return self.mode

    def _ask(self, game, legal, facts):
        """Model probabilities per direction; one-hot fallback when unavailable.

        Prefers the client's features-only entry point (option presentation,
        including label randomisation, lives in laya_client.py); otherwise
        falls back to a generic decide(state, instructions, legal).  Unknown
        probability keys are ignored: the aggregator only iterates legal moves.
        """
        client = getattr(self.decide, "__self__", None)
        if client is not None and hasattr(client, "decide_features"):
            out = client.decide_features(facts["features"], legal,
                                         meta=self._meta(game, facts))
        else:
            state, instructions = self._render_state(game, facts)
            out = self.decide(state, instructions, tuple(legal))
        probs = {}
        if isinstance(out, dict):
            move = out.get("move") or out.get("choice")
            probs = dict(out.get("probs") or {})
        elif hasattr(out, "move"):                 # Decision-style return object
            move = out.move
            probs = dict(getattr(out, "probs", None) or {})
        else:
            move = out
            last = getattr(client, "last", None)
            if isinstance(last, dict):
                probs = dict(last.get("probs") or {})
        if move not in legal:
            raise RuntimeError(f"decision client chose illegal move {move!r} from {legal}")
        if not probs:
            probs = {move: 1.0}
        return move, probs

    def _aggregate(self, legal, facts, probs):
        """Weighted blend.  With one-hot fallback probabilities (clients that
        return no distribution) engine features may override the model pick."""
        best, best_score = None, None
        for m in legal:
            f = facts["features"][m]
            pellet = 1.0 / (1.0 + f["pellet"]) if f["pellet"] < INF else 0.0
            score = (W_MODEL * probs.get(m, 0.0)
                     + W_PELLET * pellet
                     + W_EXPLORE * f["new"]
                     - W_LOOP * (1.0 if f["seen"] else 0.0)
                     - W_DANGER * DANGER_PENALTY[f["danger"]])
            if best_score is None or score > best_score:
                best, best_score = m, score
        return best

    # ------------------------------------------------------------- prompt
    def _meta(self, game, facts):
        return {"mode": self.mode, "junction": facts["here"],
                "visits": self.mem.visit_count(facts["here"]),
                "came_from": REVERSE[game.pac.dir]}

    def _render_state(self, game, facts):
        """Single source of truth: laya_client's formatter when importable."""
        try:
            from laya_client import format_features_state
        except Exception:                      # client module not available
            format_features_state = None
        if format_features_state is not None:
            state, instructions, _mapping = format_features_state(
                facts["features"], self._meta(game, facts), "labels")
            return state, instructions
        return (self._prompt_internal(game, list(facts["features"]), facts),
                INSTRUCTIONS)

    def prompt(self, game, legal, facts):
        """Compact features-only state (no board); same renderer as production."""
        return self._render_state(game, facts)[0]

    def _prompt_internal(self, game, legal, facts):
        """Fallback renderer used only when laya_client cannot be imported."""
        here = (game.pac.x, game.pac.y)
        entered_from = REVERSE[game.pac.dir]
        lines = [
            "Pac-Man at a junction. objective: 1 stay alive 2 eat pellets 3 explore new ground.",
            f"mode: {self.mode}",
            self.mem.summary(here, entered_from),
            "moves:",
        ]
        for m in legal:
            f = facts["features"][m]
            parts = [f"{m}: pellet {_fmt(f['pellet'])} zone {f['zone']} "
                     f"power {_fmt(f['power'])} danger {f['danger']}"]
            if f["edible"] is not None:
                parts.append(f"edible {_fmt(f['edible'])}")
            parts.append(f"new {_fmt_new(f['new'])}")
            if f["seen"]:
                parts.append("seen")
            lines.append(" ".join(parts))
        return "\n".join(lines)

    # ------------------------------------------------------------- entry point
    def choose(self, game, legal):
        """Full priority chain; returns (move, source)."""
        junction = (game.pac.x, game.pac.y)
        entered_from = REVERSE[game.pac.dir]
        self.mem._sync(game)               # resets first: no stale seen/loop data
        facts = self._facts(game, legal)
        self._update_mode(game, facts)
        loop = self.mem.loop_info(game, junction, entered_from)

        move = source = None
        if self._danger_fires(facts):
            move, source = self._danger_move(game, facts), "danger"
        if move is None and loop is not None:
            move, source = self.mem.escape_move(game, junction, entered_from,
                                                legal, loop), "loop"
        if move is None:
            chase = self._chase_move(game, facts, legal)
            if chase is not None:
                move, source = chase, "chase"
        if move is None:
            if self.decide is None:
                raise RuntimeError("junction reached but no decision client was given")
            model_move, probs = self._ask(game, legal, facts)
            move = self._aggregate(legal, facts, probs)
            source = "laya"

        self.mem.record(game, junction, entered_from, move)
        self.last = {"junction": junction, "entered_from": entered_from,
                     "move": move, "source": source, "loop": loop,
                     "mode": self.mode, "features": facts["features"]}
        self.history.append({"tick": self.mem.tick, "junction": junction,
                             "move": move, "source": source, "mode": self.mode,
                             "loop": None if loop is None else loop["kind"]})
        return move, source
