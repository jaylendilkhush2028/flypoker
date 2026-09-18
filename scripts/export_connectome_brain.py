#!/usr/bin/env python
"""Distil the trained real connectome brain into a compact table for the browser.

The real spiking mushroom body can't run live in a browser, and looping a
recording isn't truly random. So instead we export what the real brain *is*
after it has learned, and let the browser deal fresh random hands forever and
look its decisions up:

* ``kc_by_bin`` — for a grid of hand-strength values, the **real Kenyon cells**
  that fire (actual local indices from the connectome). The browser shows these
  firing for whatever strength a random hand happens to have.
* ``policies`` — several flies' **learned** fold/call/raise probabilities across
  that strength grid, after training a table on the real brain. Each seat plays
  one; a busted seat's replacement draws another. (Plus a naive policy.)
* ``pam_curve`` / ``ppl1_curve`` — the real PAM / PPL1 dopamine-cluster spike
  counts as a function of surprise magnitude, so wins/losses fire real dopamine.

    python scripts/export_connectome_brain.py --train 400

Writes dashboard/connectome_brain.json (~40 KB). Needs the connectome (see
scripts/run_connectome_table.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flypoker.table import Table, TableConfig
from flypoker.agent import ACTIONS
from flypoker.connectome_agent import MushroomBody, ConnectomeFlyAgent


def neutral_ctx(hs, bet, n_seats, start):
    return {"hand_strength": float(hs), "pot": 6.0, "to_call": bet,
            "n_active": n_seats, "n_seats": n_seats, "start_chips": start}


def policy_over_bins(agent, e_by_bin):
    """Fold/call/raise probabilities per bin, computed analytically.

    The connectome's KC code is deterministic for a given cue, so the only
    randomness in a decision is the softmax sampling. That means the policy is
    exactly the softmax over Q = v·e (no need to sample thousands of hands, which
    would each re-run the spiking network).
    """
    out = []
    for e in e_by_bin:
        Q = agent.v @ e
        z = np.clip(Q / agent.temperature, -30.0, 30.0)
        z -= z.max()
        p = np.exp(z); p /= p.sum()
        p = agent.explore / 3 + (1 - agent.explore) * p
        p /= p.sum()
        out.append([round(float(x), 4) for x in p])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=400, help="hands to train the table before exporting")
    ap.add_argument("--bins", type=int, default=48)
    ap.add_argument("--kc-cap", type=int, default=120)
    ap.add_argument("--seats", type=int, default=6)
    ap.add_argument("--bet", type=float, default=4.0)
    ap.add_argument("--start", type=float, default=100.0)
    ap.add_argument("--side", default="right")
    ap.add_argument("--data", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "connectome_brain.json"))
    args = ap.parse_args()

    print(">>> building the real FlyWire mushroom body…")
    # dopamine spiking isn't needed for the weight update (that uses rpe directly),
    # so keep it off for fast training and fire it only for the curves below.
    mb = MushroomBody(data_dir=args.data, side=args.side, simulate_dopamine=False, verbose=True)
    bins = np.linspace(0.02, 0.98, args.bins)
    KC_REF = ConnectomeFlyAgent(mb).kc_ref

    # real Kenyon-cell code fired at each hand strength (the shared perception).
    # The code is deterministic per cue, so one run per bin suffices.
    print(f">>> sampling the real KC code at {args.bins} hand strengths…")
    kc_by_bin, e_by_bin = [], []
    for hs in bins:
        counts = mb.kc_code(neutral_ctx(hs, args.bet, args.seats, args.start))
        e_by_bin.append(np.clip(counts / KC_REF, 0.0, 1.0))
        active = np.nonzero(counts)[0]
        if active.size > args.kc_cap:
            active = active[np.argsort(counts[active])[-args.kc_cap:]]
        kc_by_bin.append(sorted(int(k) for k in active))

    # real PAM / PPL1 spikes vs surprise magnitude (fire the real clusters here only)
    print(">>> firing the real PAM / PPL1 clusters across surprise magnitudes…")
    mags = np.linspace(0.0, 3.0, 25)
    pam_curve, ppl1_curve = [], []
    mb.cfg["simulate_dopamine"] = True
    for m in mags:
        pam, _ = mb.deliver_dopamine("reward", float(m))
        _, ppl1 = mb.deliver_dopamine("punish", float(m))
        pam_curve.append(round(pam, 1)); ppl1_curve.append(round(ppl1, 1))
    mb.cfg["simulate_dopamine"] = False

    # train a table on the real brain, then read out each fly's learned policy
    print(f">>> training a table on the real brain for {args.train} hands…")
    factory = lambda seed, name: ConnectomeFlyAgent(mb, seed=seed, name=name)
    t = Table(TableConfig(n_seats=args.seats, start_chips=args.start, ante=1, bet=args.bet,
                          seed=args.seed), agent_factory=factory)
    t.run(args.train)

    trained = sorted(t.seats, key=lambda s: s.hands, reverse=True)
    print("    seats trained (hands lived):", [s.hands for s in trained])
    policies = []
    for fly in trained:
        if fly.hands < 30:        # too green to be a useful policy
            continue
        policies.append({"hands": int(fly.hands), "p": policy_over_bins(fly, e_by_bin)})
    naive = ConnectomeFlyAgent(mb, seed=args.seed + 12345, name="naive")
    naive_p = policy_over_bins(naive, e_by_bin)

    out = {
        "meta": {
            "n_kc": mb.n_kc, "pam_n": int(len(mb.pam)), "ppl1_n": int(len(mb.ppl1)),
            "n_seats": args.seats, "start_chips": args.start, "ante": 1, "bet": args.bet,
            "kc_cap": args.kc_cap, "trained_hands": args.train, "seed": args.seed,
        },
        "bins": [round(float(b), 4) for b in bins],
        "kc_by_bin": kc_by_bin,
        "policies": policies,
        "naive_policy": naive_p,
        "pam_curve": pam_curve, "ppl1_curve": ppl1_curve,
        "mag_max": 3.0,
    }
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")))
    kb = path.stat().st_size / 1024
    print(f">>> wrote {path} ({kb:.0f} KB) · {len(policies)} trained policies · "
          f"{args.bins} strength bins")
    # quick sanity: strongest trained fly folds weak, plays strong
    if policies:
        p = policies[0]["p"]
        print(f"    top policy: weak-hand fold {p[0][0]*100:.0f}% · "
              f"strong-hand play {(p[-1][1]+p[-1][2])*100:.0f}%")


if __name__ == "__main__":
    main()
