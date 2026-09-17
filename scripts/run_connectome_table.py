#!/usr/bin/env python
"""Seat the REAL FlyWire mushroom body at the poker table.

Same game as ``run_table.py``, but every seat is the actual fruit-fly connectome
brain from FlyGambler: the poker cue drives real antennal-lobe projection neurons,
the real ALPN->KC wiring (sparsened by the APL) makes each fly's Kenyon-cell code,
and wins/losses fire the real PAM / PPL1 dopamine clusters. The fold/call/raise
readout learns on that real KC code (our dopamine-gated action-value layer).

    python scripts/run_connectome_table.py                 # 6 real-brain flies, 400 hands
    python scripts/run_connectome_table.py --hands 800 --no-dopamine   # faster (proxy affect)

Needs the `flygambler` package + its deps (brian2, scipy) importable and the
connectome data fetched (FlyGambler's scripts/download_data.py). Point the data
dir with FLYPOKER_DATA if it isn't FlyGambler's own data/. Spiking is ~0.2-0.4 s
per hand, so this is a showcase of a few hundred hands, not the thousands the
fast behavioural model runs.

Writes runs/connectome_table.json and .png (chip fortunes, real dopamine/pain +
deaths, and the policy the top fly learned on its real Kenyon cells).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flypoker.table import TableConfig
from flypoker.simulate import run_table
from flypoker.plots import plot_run


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seats", type=int, default=6)
    ap.add_argument("--hands", type=int, default=400)
    ap.add_argument("--chips", type=float, default=100.0)
    ap.add_argument("--ante", type=float, default=1.0)
    ap.add_argument("--bet", type=float, default=4.0)
    ap.add_argument("--tilt", type=float, default=0.0, help=">0: losing flies chase (tilt)")
    ap.add_argument("--side", default="right", help="brain hemisphere for the mushroom body")
    ap.add_argument("--data", default=None, help="connectome data dir (default: FlyGambler's)")
    ap.add_argument("--no-dopamine", action="store_true",
                    help="skip firing real PAM/PPL1 each hand (faster; affect via |rpe| proxy)")
    ap.add_argument("--seed", type=int, default=None, help="random each run if omitted")
    ap.add_argument("--out", default=str(ROOT / "runs" / "connectome_table.json"))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    if args.seed is None:
        args.seed = random.randrange(1_000_000)
    print(f">>> seed = {args.seed}")

    from flypoker.connectome_agent import MushroomBody, ConnectomeFlyAgent
    print(">>> building the real FlyWire mushroom body (one shared brain for the table)…")
    mb = MushroomBody(data_dir=args.data, side=args.side,
                      simulate_dopamine=not args.no_dopamine, verbose=True)

    def factory(seed, name):
        return ConnectomeFlyAgent(mb, seed=seed, name=name, tilt=args.tilt)

    cfg = TableConfig(n_seats=args.seats, start_chips=args.chips, ante=args.ante,
                      bet=args.bet, seed=args.seed)
    print(f">>> {args.seats} real-brain flies · {args.hands} hands · ante ${args.ante:g} · "
          f"bet ${args.bet:g}" + (f" · tilt {args.tilt}" if args.tilt else "")
          + ("" if args.no_dopamine else " · firing real PAM/PPL1"))
    t0 = time.time()
    table, summary = run_table(cfg, n_hands=args.hands, agent_factory=factory,
                               save=args.out, verbose=True, log_every=max(1, args.hands // 300))
    dt = time.time() - t0
    print(f"    ({dt:.0f}s total, {1000*dt/max(1,args.hands):.0f} ms/hand)")

    print(f"\n=== after {args.hands} hands on the real connectome ===")
    print(f"    deaths (flies busted & replaced): {summary['deaths']}")
    print(f"    final stacks: {summary['final_chips']}")
    name = next(iter(summary["top_policy"]))
    pol = summary["top_policy"][name]
    weak = pol[min(pol, key=lambda k: float(k))]
    strong = pol[max(pol, key=lambda k: float(k))]
    print(f"    {name} learned:  weak hand -> fold {weak['fold']*100:.0f}%   "
          f"strong hand -> play {(strong['call']+strong['raise'])*100:.0f}%")
    print(f">>> saved {args.out}")
    if not args.no_plot:
        fig = plot_run(summary, Path(args.out).with_suffix(".png"),
                       title=f"FlyPoker · REAL connectome brain · {args.seats} flies · "
                             f"{args.hands} hands · {summary['deaths']} deaths")
        print(f">>> figure: {fig}")


if __name__ == "__main__":
    main()
