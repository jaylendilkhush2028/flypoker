"""The real FlyWire connectome brain, when available, perceives and learns poker.

These tests need the `flygambler` package (+ brian2, scipy) importable and the
connectome data fetched. They **skip** cleanly otherwise, so CI without the
135 MB connectome still passes. The learning test is slow (spiking), so it is
opt-in via FLYPOKER_RUN_CONNECTOME=1.
"""

import os
from collections import Counter

import numpy as np
import pytest


def _mushroom_body():
    """Build a shared mushroom body, or skip if the connectome isn't available."""
    pytest.importorskip("flygambler", reason="FlyGambler not installed")
    pytest.importorskip("brian2", reason="brian2 not installed")
    pytest.importorskip("scipy", reason="scipy not installed")
    from flypoker.connectome_agent import MushroomBody
    try:
        return MushroomBody(simulate_dopamine=False, verbose=False)
    except (ImportError, FileNotFoundError) as e:
        pytest.skip(f"connectome unavailable: {e}")


def _ctx(hs):
    return {"hand_strength": hs, "pot": 6.0, "to_call": 4.0,
            "n_active": 6, "n_seats": 6, "start_chips": 100.0}


def test_real_kenyon_cell_code_is_sparse_and_strength_tuned():
    mb = _mushroom_body()
    weak = mb.kc_code(_ctx(0.08)) > 0
    similar = mb.kc_code(_ctx(0.12)) > 0
    strong = mb.kc_code(_ctx(0.95)) > 0

    # a real fly's KC layer is sparse — a minority of the ~2600 cells fire
    assert 0 < weak.mean() < 0.30
    # similar hands share more of their code than very different hands do
    def jac(a, b):
        return (a & b).sum() / max((a | b).sum(), 1)
    assert jac(weak, similar) > jac(weak, strong)


@pytest.mark.skipif(os.environ.get("FLYPOKER_RUN_CONNECTOME") != "1",
                    reason="slow spiking run; set FLYPOKER_RUN_CONNECTOME=1 to run")
def test_real_brain_flies_learn_to_fold_weak_and_play_strong():
    mb = _mushroom_body()
    from flypoker.connectome_agent import ConnectomeFlyAgent
    from flypoker.table import Table, TableConfig

    factory = lambda seed, name: ConnectomeFlyAgent(mb, seed=seed, name=name)
    t = Table(TableConfig(n_seats=6, start_chips=100, ante=1, bet=4, seed=1),
              agent_factory=factory)
    t.run(250)

    fly = max(t.seats, key=lambda s: s.hands)          # a well-trained survivor

    def fold_rate(hs, n=150):
        c = Counter(fly.act(_ctx(hs))[0] for _ in range(n))
        return c[0] / n                                 # action 0 == fold

    fold_weak, fold_strong = fold_rate(0.08), fold_rate(0.9)
    assert fold_weak > fold_strong + 0.25              # clearly folds weak more than strong
    assert fold_strong < 0.20                          # almost always plays strong hands
