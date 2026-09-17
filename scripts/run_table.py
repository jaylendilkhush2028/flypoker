#!/usr/bin/env python
"""Sit a table of fruit flies down to play poker, learn, get rich, and die.

    python scripts/run_table.py                 # 6 flies, 6000 hands
    python scripts/run_table.py --seats 8 --hands 12000 --tilt 0.6

Writes runs/table.json and runs/table.png (chip fortunes, dopamine/pain +
deaths, and the policy the top fly learned).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flypoker.table import TableConfig
from flypoker.simulate import run_table
from flypoker.plots import plot_run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seats", type=int, default=6)
    ap.add_argument("--hands", type=int, default=6000)
    ap.add_argument("--chips", type=float, default=100.0)
    ap.add_argument("--ante", type=float, default=1.0)
    ap.add_argument("--bet", type=float, default=4.0)
    ap.add_argument("--tilt", type=float, default=0.0, help=">0: losing flies chase (tilt)")
    ap.add_argument("--seed", type=int, default=None, help="random each run if omitted")
    ap.add_argument("--out", default=str(ROOT / "runs" / "table.json"))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    if args.seed is None:
        args.seed = random.randrange(1_000_000)
    print(f">>> seed = {args.seed}")

    cfg = TableConfig(n_seats=args.seats, start_chips=args.chips, ante=args.ante, bet=args.bet, seed=args.seed)
    print(f">>> {args.seats} flies · {args.hands} hands · ante ${args.ante:g} · bet ${args.bet:g}"
          + (f" · tilt {args.tilt}" if args.tilt else ""))
    table, summary = run_table(cfg, n_hands=args.hands,
                               agent_kwargs={"tilt": args.tilt} if args.tilt else None,
                               save=args.out, verbose=True)

    print(f"\n=== after {args.hands} hands ===")
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
        fig = plot_run(summary, Path(args.out).with_suffix(".png"))
        print(f">>> figure: {fig}")


if __name__ == "__main__":
    main()
