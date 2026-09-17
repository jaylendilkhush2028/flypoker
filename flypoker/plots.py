"""A three-panel figure of a FlyPoker run."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_run(summary: dict, path: str | Path, title: str | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = summary["history"]
    hand = np.array(h["hand"])
    chips = np.array(h["chips"])            # (T, n_seats)
    n_seats = chips.shape[1] if chips.size else 0

    fig, ax = plt.subplots(3, 1, figsize=(11, 11))
    fig.suptitle(title or f"FlyPoker · {summary['config']['n_seats']} flies · "
                 f"{summary['config']['n_hands']} hands · {summary['deaths']} deaths",
                 fontsize=14, fontweight="bold")

    # (1) chip fortunes per seat
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, max(1, n_seats)))
    for s in range(n_seats):
        ax[0].plot(hand, chips[:, s], lw=1.3, color=palette[s], alpha=.9)
    ax[0].axhline(0, color="#d1495b", ls="--", lw=1)
    ax[0].set_ylabel("chips"); ax[0].set_xlabel("hand")
    ax[0].set_title("Chip fortunes — each line a seat; drops to ~0 are deaths (a new fly buys in)",
                    loc="left", fontsize=10)

    # (2) dopamine / pain + cumulative deaths
    def roll(x, w=15):
        x = np.asarray(x, float)
        if len(x) < 2:
            return x
        w = max(1, min(w, len(x)))
        return np.convolve(x, np.ones(w) / w, mode="same")
    ax[1].plot(hand, roll(h["dopamine"]), color="#2a9d8f", lw=1.5, label="dopamine (won)")
    ax[1].plot(hand, roll(h["pain"]), color="#d1495b", lw=1.5, label="pain (lost)")
    ax[1].set_ylabel("avg per fly / hand"); ax[1].set_xlabel("hand")
    ax[1].legend(loc="upper left", fontsize=8)
    axd = ax[1].twinx()
    axd.plot(hand, h["deaths"], color="#8d99ae", lw=1.4, ls=":")
    axd.set_ylabel("cumulative deaths", color="#8d99ae")
    ax[1].set_title("Dopamine & pain across the table, and how many flies have died",
                    loc="left", fontsize=10)

    # (3) learned policy of the top surviving fly
    pol = summary.get("top_policy", {})
    if pol:
        name = next(iter(pol))
        p = pol[name]
        xs = sorted(float(k) for k in p)
        fold = [p[k]["fold"] for k in map(_key, xs, [p] * len(xs))]
        call = [p[k]["call"] for k in map(_key, xs, [p] * len(xs))]
        rais = [p[k]["raise"] for k in map(_key, xs, [p] * len(xs))]
        ax[2].stackplot(xs, fold, call, rais,
                        labels=["fold", "call", "raise"],
                        colors=["#8d99ae", "#4aa3ff", "#f4a261"], alpha=.9)
        ax[2].set_xlim(min(xs), max(xs)); ax[2].set_ylim(0, 1)
        ax[2].set_xlabel("hand strength (weak → strong)")
        ax[2].set_ylabel("P(action)")
        ax[2].legend(loc="upper center", ncol=3, fontsize=8)
        ax[2].set_title(f"What {name} learned — fold weak hands, play strong ones (no rules given)",
                        loc="left", fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _key(x, p):
    """Map a float hand-strength back to its dict key (keys are rounded floats)."""
    for k in p:
        if abs(float(k) - x) < 1e-6:
            return k
    return min(p, key=lambda k: abs(float(k) - x))
