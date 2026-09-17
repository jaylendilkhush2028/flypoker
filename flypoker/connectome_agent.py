"""The *real* fruit-fly connectome, seated at the poker table.

``FlyAgent`` (in ``agent.py``) is a fast behavioural stand-in: a random
projection plays the part of the Kenyon-cell layer. This module replaces that
stand-in with the **actual FlyWire mushroom-body circuit** from the FlyGambler
project, so each seat is literally the real fly brain deciding fold / call /
raise.

What runs on the real connectome (via ``flygambler``):

* **Perception.** The poker cue (hand strength + pot odds + table context) is
  injected as drive onto real antennal-lobe projection neurons (ALPNs). The real
  ALPN→KC wiring, sparsened by the APL feedback neuron, produces a sparse
  **Kenyon-cell code** — ~1–2% of ~2,600 real KCs, exactly as in a living fly.
  This code *is* the fly's representation of the situation.
* **Affect.** After the hand, winning fires the real **PAM** reward-dopamine
  cluster (153 neurons) and losing fires the real **PPL1** punishment cluster
  (8 neurons). The dopamine / pain we report are those clusters' actual spikes.

What is our added layer (honestly flagged, exactly as FlyGambler flags its
plasticity): the published whole-brain model has fixed weights and does not
learn. We add a **dopamine-gated, action-value readout** from the real Kenyon
cells to the three poker actions. It is trained with the same three-factor rule
(KC eligibility × dopamine-signed error), and it is action-value rather than a
single approach/avoid drive so a table of self-interested flies can't collapse
to "always call" (each action converges to *its own* expected payoff).

One shared :class:`MushroomBody` (one connectome, one engine) backs the whole
table; every fly perceives through the same real wiring and differs only in what
it has learned — clonal flies with different experience. Build it once, then
hand a :class:`ConnectomeFlyAgent` factory to :class:`flypoker.table.Table`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ACTIONS = ("fold", "call", "raise")


def _default_data_dir() -> Path:
    """Locate the connectome data: env override, else FlyGambler's own data dir."""
    import os
    env = os.environ.get("FLYPOKER_DATA") or os.environ.get("FLYGAMBLER_DATA")
    if env:
        return Path(env)
    try:
        import flygambler
        cand = Path(flygambler.__file__).resolve().parents[1] / "data"
        if cand.exists():
            return cand
    except Exception:
        pass
    return Path("data")


class MushroomBody:
    """The shared, real FlyWire mushroom-body engine for a whole table.

    Wraps FlyGambler's connectome loader + fast LIF engine. Holds the cue
    encoder (feature → ALPN drive) and exposes ``kc_code`` (real sparse Kenyon
    activity for a poker cue) and ``deliver_dopamine`` (fire real PAM / PPL1).
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        side: str = "right",
        n_channels: int = 16,
        hs_channels: int = 12,
        hs_sigma: float = 0.12,
        cue_drive_mV: float = 55.0,
        t_cue_ms: float = 80.0,
        dt_ms: float = 0.5,
        simulate_dopamine: bool = True,
        t_teach_ms: float = 40.0,
        dan_drive_mV: float = 45.0,
        seed: int = 0,
        verbose: bool = True,
    ):
        try:
            from flygambler.connectome import load_tables, extract_circuit, mushroom_body_ids
            from flygambler.fastlif import FastLIF
        except Exception as e:  # pragma: no cover - environment guard
            raise ImportError(
                "The connectome brain needs the `flygambler` package and its deps "
                "(brian2, scipy). Install FlyGambler into this environment, e.g.\n"
                "    pip install -e /path/to/FlyGambler\n"
                "and fetch the connectome once with FlyGambler's "
                "`python scripts/download_data.py`.\n"
                f"(underlying import error: {e})"
            ) from e

        data_dir = Path(data_dir) if data_dir is not None else _default_data_dir()
        if not (data_dir / "Connectivity_783.parquet").exists():
            raise FileNotFoundError(
                f"Connectome data not found in {data_dir}. Point FLYPOKER_DATA at "
                "FlyGambler's data/ dir, or run FlyGambler's scripts/download_data.py."
            )

        self.cfg = dict(
            n_channels=n_channels, hs_channels=hs_channels, hs_sigma=hs_sigma,
            cue_drive_mV=cue_drive_mV, t_cue_ms=t_cue_ms, t_teach_ms=t_teach_ms,
            dan_drive_mV=dan_drive_mV, simulate_dopamine=simulate_dopamine,
        )
        rng = np.random.default_rng(seed)
        tables = load_tables(data_dir)
        ids = mushroom_body_ids(tables, side=side)
        circuit = extract_circuit(tables, ids, verbose=verbose)
        self.engine = FastLIF(circuit, dt_ms=dt_ms)

        self.alpn = self.engine.role_idx["ALPN"]
        self.kc = self.engine.role_idx["KC"]
        self.pam = self.engine.role_idx["PAM"]
        self.ppl1 = self.engine.role_idx["PPL1"]
        self.n_kc = int(len(self.kc))
        self.n_neurons = int(self.engine.n_neurons)

        # partition ALPNs into feature channels (real neurons carry the cue)
        self.channels = np.array_split(rng.permutation(self.alpn), n_channels)
        # overlapping receptive fields tiling hand strength in [0, 1]
        self.hs_centers = np.linspace(0.05, 0.95, hs_channels)
        if verbose:
            print(f"    mushroom body ready: {self.n_kc} KC, "
                  f"{len(self.engine.role_idx['MBON'])} MBON, "
                  f"{len(self.pam)} PAM, {len(self.ppl1)} PPL1")

    # ------------------------------------------------------------------ cue
    def _channel_activation(self, ctx: dict) -> np.ndarray:
        c = self.cfg
        hs = float(ctx["hand_strength"])
        pot, to_call = ctx["pot"], ctx["to_call"]
        pot_odds = to_call / (pot + to_call + 1e-9)
        a = np.zeros(c["n_channels"])
        nh = c["hs_channels"]
        a[:nh] = np.exp(-((hs - self.hs_centers) ** 2) / (2 * c["hs_sigma"] ** 2))
        extra = [
            pot_odds,                                   # price to stay in
            ctx["n_active"] / ctx["n_seats"],           # how many opponents
            min(1.0, ctx.get("stack_frac", 1.0)),       # stack depth
            1.0,                                        # tonic bias channel
        ]
        for j, val in enumerate(extra[: c["n_channels"] - nh]):
            a[nh + j] = float(val)
        return a

    def kc_code(self, ctx: dict) -> np.ndarray:
        """Real sparse Kenyon-cell spike counts for this poker cue (len n_kc)."""
        act = self._channel_activation(ctx)
        drive = np.zeros(self.n_neurons)
        for ch, val in zip(self.channels, act):
            drive[ch] = val * self.cfg["cue_drive_mV"]
        self.engine.reset_state()
        self.engine.set_stim_vector(drive)
        counts = self.engine.run_counts(self.cfg["t_cue_ms"])
        return counts[self.kc].astype(np.float64)

    def deliver_dopamine(self, valence: str, magnitude: float) -> tuple[float, float]:
        """Fire the real PAM (reward) or PPL1 (punishment) cluster; return spikes."""
        if not self.cfg["simulate_dopamine"]:
            return (magnitude, 0.0) if valence == "reward" else (0.0, magnitude)
        target = self.pam if valence == "reward" else self.ppl1
        drive = np.zeros(self.n_neurons)
        drive[target] = min(magnitude, 1.5) * self.cfg["dan_drive_mV"]
        self.engine.reset_state()
        self.engine.set_stim_vector(drive)
        d = self.engine.run_counts(self.cfg["t_teach_ms"])
        return float(d[self.pam].sum()), float(d[self.ppl1].sum())


class ConnectomeFlyAgent:
    """A poker fly whose perception and dopamine are the real connectome.

    Drop-in for :class:`flypoker.agent.FlyAgent`: same ``act`` / ``learn`` API
    and life-stat fields, so :class:`flypoker.table.Table` runs it unchanged
    (pass an ``agent_factory``). All flies share one :class:`MushroomBody`; each
    carries its own learned readout ``v`` (Kenyon cells → action value).
    """

    def __init__(
        self,
        mb: MushroomBody,
        *,
        lr: float = 0.15,
        temperature: float = 0.35,
        explore: float = 0.03,
        kc_ref: float = 3.0,
        decay: float = 0.0,
        mood_gain: float = 0.15,
        mood_recovery: float = 0.97,
        tilt: float = 0.0,
        seed: int = 0,
        name: str = "fly",
    ):
        self.mb = mb
        self.rng = np.random.default_rng(seed)
        self.lr = lr
        self.temperature = temperature
        self.explore = explore
        self.kc_ref = kc_ref
        self.decay = decay
        self.mood_gain = mood_gain
        self.mood_recovery = mood_recovery
        self.tilt = tilt
        self.name = name

        self.v = np.zeros((3, mb.n_kc))     # learned KC -> action-value readout
        self.mood = 0.0

        # life stats (set by the table when a fly is seated)
        self.chips = 0.0
        self.alive = True
        self.hands = 0
        self.wins = 0
        self.born_hand = 0

    def _eligibility(self, kc_counts: np.ndarray) -> np.ndarray:
        # graded eligibility of the responsive Kenyon cells (the APL already
        # sparsened them upstream); clip so a very active KC saturates.
        return np.clip(kc_counts / self.kc_ref, 0.0, 1.0)

    def act(self, ctx: dict):
        ctx = dict(ctx)
        ctx.setdefault("stack_frac", min(1.0, self.chips / ctx["start_chips"]))
        e = self._eligibility(self.mb.kc_code(ctx))
        Q = self.v @ e                                   # value of fold/call/raise
        if self.tilt:                                    # sad -> lean to action (chasing)
            Q = Q + np.array([-1.0, 0.0, 1.0]) * self.tilt * max(0.0, -self.mood)
        z = np.clip(Q / self.temperature, -30.0, 30.0)
        z -= z.max()
        p = np.exp(z); p /= p.sum()
        p = self.explore / 3 + (1 - self.explore) * p
        a = int(self.rng.choice(3, p=p / p.sum()))
        return a, {"e": e, "Q": Q}

    def learn(self, ctx_learn: dict, action_idx: int, reward: float):
        """Dopamine-gated action-value update on the real Kenyon-cell code.

        ``reward`` is the hand's chip delta in bet-units. The error is measured
        against the value of the action actually taken, so fold / call / raise
        each converge to their own expected payoff for this kind of hand — which
        is what teaches the fly to fold trash and press quality.
        """
        self.hands += 1
        e = ctx_learn["e"]
        Q = ctx_learn["Q"]
        rpe = reward - float(Q[action_idx])
        norm = float(e.sum()) + 1e-9                     # keep ΔQ ≈ lr * rpe
        self.v[action_idx] += (self.lr / norm) * rpe * e
        if self.decay:                                   # slow forgetting toward naive
            self.v[action_idx] *= (1.0 - self.decay)

        self.mood = self.mood_recovery * self.mood + self.mood_gain * (
            reward if reward < 0 else 0.6 * reward)
        self.mood = float(np.clip(self.mood, -4.0, 3.0))

        valence = "reward" if rpe >= 0 else "punish"
        pam, ppl1 = self.mb.deliver_dopamine(valence, abs(rpe))
        return {"rpe": rpe, "dopamine": pam, "pain": ppl1, "mood": self.mood}
