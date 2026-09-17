"""Run a table for many hands, track the ecology, and measure what flies learn."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .table import Table, TableConfig
from .agent import ACTIONS


def policy_probe(agent, hand_strengths=None, n=400, n_seats=6, start_chips=100.0, bet=4.0):
    """Empirical fold/call/raise mix of an agent across hand strengths."""
    hand_strengths = hand_strengths if hand_strengths is not None else np.linspace(0.05, 0.95, 10)
    out = {}
    for hs in hand_strengths:
        ctx = {"hand_strength": float(hs), "pot": 6.0, "to_call": bet,
               "n_active": n_seats, "n_seats": n_seats, "start_chips": start_chips}
        c = Counter(ACTIONS[agent.act(ctx)[0]] for _ in range(n))
        out[round(float(hs), 3)] = {a: c[a] / n for a in ACTIONS}
    return out


def run_table(cfg: TableConfig | None = None, n_hands: int = 6000, agent_kwargs: dict | None = None,
              log_every: int = 10, save: str | Path | None = None, verbose: bool = False):
    """Run the table, sampling chip stacks / dopamine / pain / deaths over time."""
    cfg = cfg or TableConfig()
    table = Table(cfg, agent_kwargs=agent_kwargs)
    hist = {"hand": [], "chips": [], "deaths": [], "dopamine": [], "pain": [],
            "played_strength": [], "folded_strength": []}

    def on_hand(rec):
        if rec["hand"] % log_every:
            return
        chips = [0.0] * cfg.n_seats
        dop = pain = 0.0
        played = []
        folded = []
        for s in rec["seats"]:
            chips[s["seat"]] = s["chips"]
            dop += s["dopamine"]; pain += s["pain"]
            (folded if s["action"] == "fold" else played).append(s["hand_strength"])
        hist["hand"].append(rec["hand"])
        hist["chips"].append(chips)
        hist["deaths"].append(rec["deaths_total"])
        hist["dopamine"].append(dop / max(1, len(rec["seats"])))
        hist["pain"].append(pain / max(1, len(rec["seats"])))
        hist["played_strength"].append(float(np.mean(played)) if played else np.nan)
        hist["folded_strength"].append(float(np.mean(folded)) if folded else np.nan)

    table.run(n_hands, on_hand=on_hand)

    survivors = sorted(table.seats, key=lambda s: s.hands, reverse=True)
    policies = {s.name: policy_probe(s, n_seats=cfg.n_seats, start_chips=cfg.start_chips, bet=cfg.bet)
                for s in survivors[:3]}
    summary = {
        "config": {"n_seats": cfg.n_seats, "start_chips": cfg.start_chips, "ante": cfg.ante,
                   "bet": cfg.bet, "seed": cfg.seed, "n_hands": n_hands},
        "deaths": table.deaths,
        "final_chips": {s.name: round(s.chips, 1) for s in table.seats},
        "top_policy": policies,
        "history": hist,
    }
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        Path(save).write_text(json.dumps(summary))
    if verbose:
        print(f"  {n_hands} hands · {table.deaths} deaths · "
              f"stacks {sorted([round(s.chips) for s in table.seats], reverse=True)}")
    return table, summary
