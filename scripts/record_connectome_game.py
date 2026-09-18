#!/usr/bin/env python
"""Play a game on the REAL connectome and record it for the 3D replay viewer.

Runs a table of real-FlyWire-mushroom-body flies (see ``connectome_agent.py``)
and writes, for every hand and every fly, exactly what the real brain did: its
hole cards, the **actual Kenyon cells that fired** (real local indices), the
fold/call/raise values it read out, the action it took, and the **real PAM /
PPL1 dopamine spikes** the outcome triggered. ``dashboard/connectome_replay.html``
loads this log and plays it back with the 3D table + brain panel — so what you
watch on screen is the genuine spiking brain's decisions, not the JS stand-in.

    python scripts/record_connectome_game.py --hands 60 --seed 7

Writes dashboard/connectome_game.json. Needs the connectome available (see
scripts/run_connectome_table.py). ~0.2-0.4 s per hand (spiking).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flypoker.cards import Deck, best_of_seven, hand_strength
from flypoker.agent import ACTIONS
from flypoker.connectome_agent import MushroomBody, ConnectomeFlyAgent

FLY_NAMES = ["Buzz", "Zip", "Dart", "Gnat", "Midge", "Vex", "Pip", "Skit", "Blip", "Fizz"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hands", type=int, default=60)
    ap.add_argument("--seats", type=int, default=6)
    ap.add_argument("--chips", type=float, default=100.0)
    ap.add_argument("--ante", type=float, default=1.0)
    ap.add_argument("--bet", type=float, default=4.0)
    ap.add_argument("--kc-cap", type=int, default=120, help="max KC indices logged per decision")
    ap.add_argument("--side", default="right")
    ap.add_argument("--data", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "connectome_game.json"))
    args = ap.parse_args()

    print(">>> building the real FlyWire mushroom body…")
    mb = MushroomBody(data_dir=args.data, side=args.side, simulate_dopamine=True, verbose=True)
    rng = random.Random(args.seed)

    next_id = 0
    def new_fly(seat):
        nonlocal next_id
        f = ConnectomeFlyAgent(mb, seed=args.seed * 1000 + next_id,
                               name=f"{FLY_NAMES[seat % len(FLY_NAMES)]}#{next_id}")
        f.chips = args.chips; f.seat = seat; next_id += 1
        return f

    seats = [new_fly(i) for i in range(args.seats)]
    ante, bet = args.ante, args.bet
    hands_log = []
    deaths = 0

    print(f">>> recording {args.hands} hands on the real connectome…")
    t0 = time.time()
    for hno in range(1, args.hands + 1):
        newborn = []
        for i, fly in enumerate(seats):
            if fly.chips < ante:
                deaths += 1; seats[i] = new_fly(i); newborn.append(i)

        button = (hno - 1) % args.seats
        deck = Deck(random.Random(rng.random())).shuffle()
        community = deck.draw(5)
        holes = {i: deck.draw(2) for i in range(args.seats)}

        pot = 0.0; committed = {}
        for i in range(args.seats):
            seats[i].chips -= ante; committed[i] = ante; pot += ante

        rec_seats = []
        decisions = {}; learn_ctx = {}
        for i in range(args.seats):
            fly = seats[i]
            score = best_of_seven(holes[i] + community)
            hs = hand_strength(score)
            ctx = {"hand_strength": hs, "pot": pot, "to_call": bet,
                   "n_active": args.seats, "n_seats": args.seats, "start_chips": args.chips}
            chips0 = fly.chips
            a_idx, lc = fly.act(ctx)
            want = (0.0, bet, 2 * bet)[a_idx]
            put = min(want, fly.chips)
            fly.chips -= put; committed[i] += put; pot += put
            decisions[i] = (a_idx, put); learn_ctx[i] = lc
            # the real Kenyon cells that fired (cap to the strongest for a tidy log)
            e = lc["e"]
            active = np.nonzero(e)[0]
            if active.size > args.kc_cap:
                active = active[np.argsort(e[active])[-args.kc_cap:]]
            rec_seats.append({
                "seat": i, "name": fly.name,
                "hole": [list(holes[i][0]), list(holes[i][1])],
                "hs": round(float(hs), 3), "chips0": round(float(chips0), 1),
                "action": ACTIONS[a_idx], "put": round(float(put), 1),
                "Q": [round(float(q), 3) for q in lc["Q"]],
                "kc": sorted(int(k) for k in active),
            })

        contenders = [i for i in range(args.seats)
                      if decisions[i][0] != 0 and committed[i] > ante]
        if contenders:
            best = max(best_of_seven(holes[i] + community) for i in contenders)
            winners = [i for i in contenders if best_of_seven(holes[i] + community) == best]
        else:
            winners = list(range(args.seats))
        share = pot / len(winners) if winners else 0.0
        for i in winners:
            seats[i].chips += share

        for i in range(args.seats):
            fly = seats[i]
            a_idx, put = decisions[i]
            delta = (share if i in winners else 0.0) - committed[i]
            fb = fly.learn(learn_ctx[i], a_idx, delta / bet)
            fly.wins += 1 if (i in winners and a_idx != 0) else 0
            rs = rec_seats[i]
            rs.update({
                "pam": round(float(fb["dopamine"]), 1), "ppl1": round(float(fb["pain"]), 1),
                "chips1": round(float(fly.chips), 1), "delta": round(float(delta), 1),
                "won": i in winners, "folded": a_idx == 0,
            })

        hands_log.append({
            "h": hno, "button": button,
            "community": [list(c) for c in community],
            "pot": round(float(pot), 1),
            "newborn": newborn,
            "winners": winners,
            "seats": rec_seats,
        })
        if hno % 10 == 0:
            print(f"    hand {hno}/{args.hands}  ({(time.time()-t0):.0f}s, {deaths} deaths)")

    out = {
        "meta": {
            "n_seats": args.seats, "start_chips": args.chips, "ante": ante, "bet": bet,
            "n_kc": mb.n_kc, "kc_cap": args.kc_cap,
            "pam_n": int(len(mb.pam)), "ppl1_n": int(len(mb.ppl1)),
            "hands": args.hands, "seed": args.seed, "deaths": deaths,
            "names": FLY_NAMES[: args.seats],
        },
        "hands": hands_log,
    }
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")))
    kb = path.stat().st_size / 1024
    print(f">>> wrote {path} ({kb:.0f} KB) · {args.hands} hands · {deaths} deaths · "
          f"{(time.time()-t0):.0f}s")


if __name__ == "__main__":
    main()
