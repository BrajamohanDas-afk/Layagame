"""Thin wrapper over the local Laya decision model for Pac-Man's junctions.

The model is loaded in-process on CUDA (no sidecar, no HTTP, no MCP).  There
is no CPU fallback by design: if the GPU, the checkpoint or the prompt
contract is unavailable, this raises with the exact fix.

Presentation modes (WP3 anti-bias experiment; deterministic and cache-safe):
  labels    criteria keys are the real directions (baseline behaviour)
  shuffled  criteria keys are A/B/C/D and the deterministic mapping is shown
            to the model ("A: move right")
  neutral   criteria keys are A/B/C/D with neutral descriptions; the mapping
            is hidden from the model and known only to this client

Every mode returns direction-keyed probabilities to the caller, so the
aggregator never sees option letters.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import OrderedDict

DEFAULT_MODEL = os.environ.get(
    "LAYA_MODEL",
    r"C:\Users\Braju\.cache\huggingface\hub\models--convaiinnovations--laya"
    r"\snapshots\55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
)

# Labels must stay short: long option descriptions are silently re-cut
# (~45 tokens/label ceiling) and the model would see less than asked.
DIRECTION_LABEL = {"up": "move up", "down": "move down",
                   "left": "move left", "right": "move right"}
DIR_ORDER = ("up", "down", "left", "right")
OPTION_KEYS = "ABCD"
PRESENTATION_MODES = ("labels", "shuffled", "neutral")
MAPPING_VERSION = 1        # bump when the mapping function changes
PRESENTATION_VERSION = 1   # bump when labels/instructions wording changes
SUM_ROUNDING_TOLERANCE = 0.02
MAX_CACHE = 4096
UNREACHABLE = 1 << 30   # matches brain.INF: render as "-" in the prompt


def canonical_legal(legal) -> tuple:
    """Dedupe and order moves canonically so caller order never reaches a key."""
    wanted = set(legal)
    unknown = wanted - set(DIR_ORDER)
    if unknown:
        raise RuntimeError(f"unknown move(s) {sorted(unknown)}; expected {DIR_ORDER}")
    return tuple(d for d in DIR_ORDER if d in wanted)


def option_mapping(state: str, mode: str, legal) -> dict:
    """Deterministic option-key -> direction mapping, stable across processes.

    labels mode maps directions to themselves.  shuffled/neutral rank each
    direction by a sha256 digest of (version, mode, state, direction): the
    relative order of surviving directions is preserved when one is removed,
    but letter keys shift (the cache key always includes the mapping).  Never
    uses hash()/id()/global RNG.
    """
    dirs = canonical_legal(legal)
    if mode == "labels":
        return {d: d for d in dirs}

    def rank(d: str) -> bytes:
        payload = json.dumps([MAPPING_VERSION, mode, state, d],
                             ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).digest()

    ordered = sorted(dirs, key=lambda d: (rank(d), d))
    return dict(zip(OPTION_KEYS, ordered))


def _d(value):
    """Render unreachable distances as '-' (power runs out mid-level)."""
    return "-" if isinstance(value, int) and value >= UNREACHABLE else value


def _feature_line(name: str, f: dict) -> str:
    parts = [f"pellet {_d(f.get('pellet', '?'))}",
             f"zone {f.get('zone', '?')}",
             f"power {_d(f.get('power', '?'))}",
             f"danger {f.get('danger', '?')}"]
    if f.get("edible") is not None:
        parts.append(f"edible {_d(f['edible'])}")
    if f.get("new") is not None:
        parts.append(f"new {float(f['new']):.1f}")
    if f.get("seen"):
        parts.append("seen")
    return f"{name}: " + " ".join(parts)


def format_features_state(features: dict, meta: dict | None = None,
                          mode: str = "labels", mapping: dict | None = None):
    """Render the compact features-only state (no board) and its instructions.

    `features` is the engine contract: {direction: {pellet, zone, power,
    danger, edible, new, seen}}.  In shuffled/neutral modes the move lines are
    keyed by option letters; in neutral mode no direction word is emitted.
    Returns (state_text, instructions, mapping) — the mapping MUST be passed
    to decide() so the letters in the text and the translation agree.
    """
    meta = meta or {}
    legal = canonical_legal(features.keys())
    dir_to_key = None
    if mapping is None:
        # seed the mapping from the direction-keyed rendering (stable per state)
        canonical = _render_features(features, meta, None)
        mapping = option_mapping(canonical, mode, legal)
    if mode != "labels":
        dir_to_key = {d: k for k, d in mapping.items()}

    lines = ["Pac-Man at a junction. objective: 1 stay alive 2 eat pellets 3 explore new ground."]
    if meta.get("mode"):
        lines.append(f"mode: {meta['mode']}")
    junction = meta.get("junction")
    if junction is not None:
        line = f"junction J{tuple(junction)}"
        if meta.get("visits"):
            line += f" visited {meta['visits']}x"
        came = meta.get("came_from")
        if came and dir_to_key is None:  # letter modes: a direction word would leak
            line += f", came from {came}"
        lines.append(line)
    lines.append("moves:")
    for d in legal:
        key = dir_to_key[d] if dir_to_key else d
        lines.append(_feature_line(key, features[d]))
    state = "\n".join(lines)

    if mode == "labels":
        instructions = "Pick the best direction for Pac-Man."
    elif mode == "shuffled":
        instructions = "Options: " + ", ".join(f"{k} = move {d}" for k, d in mapping.items()) + "."
    else:
        instructions = "Options: " + ", ".join(mapping) + ". Pick the best option."
    return state, instructions, mapping


def _render_features(features: dict, meta: dict, mapping: dict | None) -> str:
    dir_to_key = {d: k for k, d in mapping.items()} if mapping else None
    lines = ["Pac-Man at a junction."]
    if meta.get("junction") is not None:
        lines.append(f"junction J{tuple(meta['junction'])}")
    for d in canonical_legal(features.keys()):
        key = dir_to_key[d] if dir_to_key else d
        lines.append(_feature_line(key, features[d]))
    return "\n".join(lines)


class LayaClient:
    """decide(state, instructions, legal) -> one of the legal moves.

    `agent` may be injected for tests; the real model is only loaded when
    agent is None.  `mode` selects the option presentation (see module doc).
    """

    def __init__(self, model: str | None = None, device: str = "cuda",
                 mode: str = "labels", agent=None):
        if mode not in PRESENTATION_MODES:
            raise ValueError(f"unknown presentation mode {mode!r}; pick one of {PRESENTATION_MODES}")
        self.mode = mode
        self.cache: OrderedDict = OrderedDict()
        self.calls = 0
        self.last = None
        if agent is not None:
            self.agent = agent
            self.device = getattr(agent, "device", None)
            return

        model = model or DEFAULT_MODEL
        try:
            import torch
            import laya
        except ImportError as e:
            raise RuntimeError(
                "Laya/torch are not installed in this environment. "
                "Run: uv sync --group gpu"
            ) from e
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available: the RTX 3050 is powered down or this torch build "
                "is CPU-only. Plug in AC power, check `nvidia-smi`, then run "
                "`uv sync --group gpu`. CPU play is out of scope by design."
            )
        if not os.path.exists(model):
            raise RuntimeError(
                f"Laya checkpoint not found at {model!r}. Point LAYA_MODEL at a local "
                "convaiinnovations/laya snapshot (fully offline) or download it once."
            )
        self.agent = laya.load(model, device=device)
        if getattr(self.agent.device, "type", None) != "cuda":
            raise RuntimeError(
                f"Laya fell back to {self.agent.device} during load; the game requires CUDA."
            )
        self.agent.warmup()

    # ------------------------------------------------------------------ decide

    def decide(self, state: str, instructions: str, legal,
               mode: str | None = None, mapping: dict | None = None) -> str:
        mode = mode or self.mode
        if mode not in PRESENTATION_MODES:
            raise ValueError(f"unknown presentation mode {mode!r}; pick one of {PRESENTATION_MODES}")
        legal = canonical_legal(legal)
        if not legal:
            raise RuntimeError("no legal moves to decide")
        if len(legal) == 1:
            return legal[0]  # forced move: no model call, no cache entry

        if mapping is None:
            mapping = option_mapping(state, mode, legal)
        elif set(mapping.values()) != set(legal):
            raise RuntimeError(f"mapping {mapping} does not cover the legal moves {legal}")
        key = self._cache_key(state, instructions, legal, mode, mapping)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache.move_to_end(key)
            self._record(hit, cached=True)
            print(f"[laya] cache hit mode={mode} legal={','.join(legal)} -> {hit['chosen']}", flush=True)
            return hit["chosen"]

        criteria = {k: (DIRECTION_LABEL[d] if mode != "neutral" else f"option {k}")
                    for k, d in mapping.items()}
        questions = {"move": {"type": "choice", "instructions": instructions,
                              "criteria": criteria}}
        t0 = time.perf_counter()
        out = self.agent.predict(state, questions)
        ms = (time.perf_counter() - t0) * 1000.0
        usage = out.get("usage") or {}
        answer = (out.get("answers") or {}).get("move")
        if answer is None:
            raise RuntimeError(f"Laya returned no 'move' answer: {out!r}")
        raw = answer.get("probabilities") or {}

        if usage.get("truncated"):
            raise RuntimeError(
                "Laya truncated the state (state_tokens="
                f"{usage.get('state_tokens')}, dropped={usage.get('state_tokens_dropped')}). "
                "The model must see the whole state, every time."
            )
        if usage.get("options"):
            raise RuntimeError(
                f"Laya collapsed question options: {usage['options']}. "
                "Keep option labels short ('move up')."
            )
        if set(raw) != set(mapping):
            raise RuntimeError(
                f"Laya returned options {sorted(raw)} but {sorted(mapping)} were presented."
            )
        if not all(math.isfinite(v) and v >= 0 for v in raw.values()):
            raise RuntimeError(f"Laya returned non-finite probabilities: {raw!r}")
        total = sum(raw.values())
        if abs(total - 1.0) > SUM_ROUNDING_TOLERANCE:
            raise RuntimeError(f"Laya probabilities do not sum to ~1.0 ({total:.3f}): {raw!r}")

        probs = {d: raw[k] for k, d in mapping.items()}          # direction-keyed
        chosen = max(legal, key=lambda d: probs[d])              # canonical tie-break
        entry = {"chosen": chosen, "probs": dict(probs), "raw": dict(raw),
                 "mapping": dict(mapping), "mode": mode, "ms": ms,
                 "usage": {"truncated": usage.get("truncated"),
                           "options": usage.get("options"),
                           "state_tokens": usage.get("state_tokens"),
                           "input_tokens": usage.get("input_tokens")}}
        self.cache[key] = entry
        if len(self.cache) > MAX_CACHE:
            self.cache.popitem(last=False)
        self.calls += 1
        self._record(entry, cached=False)
        pretty = ", ".join(f"{k}={v:.2f}" for k, v in raw.items())
        print(f"[laya] mode={mode} legal={','.join(legal)} [{pretty}] -> {chosen} ({ms:.0f} ms)", flush=True)
        return chosen

    def decide_features(self, features: dict, legal, meta: dict | None = None,
                        mode: str | None = None) -> str:
        """Features-only entry point: the engine passes feature dicts, not text."""
        mode = mode or self.mode
        if mode not in PRESENTATION_MODES:
            raise ValueError(f"unknown presentation mode {mode!r}; pick one of {PRESENTATION_MODES}")
        state, instructions, mapping = format_features_state(features, meta, mode)
        legal = canonical_legal(legal)
        return self.decide(state, instructions, legal, mode=mode, mapping=mapping)

    # ------------------------------------------------------------------ cache

    def _cache_key(self, state, instructions, legal, mode, mapping):
        return (PRESENTATION_VERSION, mode, state, instructions, tuple(legal),
                tuple((k, mapping[k]) for k in mapping))

    def _record(self, entry: dict, cached: bool):
        key_of = {d: k for k, d in entry["mapping"].items()}
        keys = list(entry["mapping"])
        self.last = {
            "legal": tuple(entry["probs"]),
            "probs": dict(entry["probs"]),
            "raw": dict(entry["raw"]),
            "choice": entry["chosen"],
            "position": keys.index(key_of[entry["chosen"]]),
            "mapping": dict(entry["mapping"]),
            "mode": entry["mode"],
            "ms": entry["ms"],
            "cached": cached,
            "truncated": entry["usage"].get("truncated"),
            "options": entry["usage"].get("options"),
            "state_tokens": entry["usage"].get("state_tokens"),
        }

    @staticmethod
    def mapping_for(state: str, legal, mode: str = "labels") -> dict:
        return option_mapping(state, mode, legal)
