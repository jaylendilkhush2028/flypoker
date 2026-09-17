"""A poker-playing fruit fly: the mushroom-body reward-learning brain.

Same base idea as FlyGambler, adapted to a 3-way poker decision. Each hand the
fly perceives a **cue** (its hand strength + table context), the cue is expanded
into a sparse **Kenyon-cell** code (a random projection + k-winner-take-all, as
the real mushroom body does), and dopamine-gated weights read out an action
value for **fold / call / raise**. It picks an action by softmax.

Afterwards the chip result drives learning: winning fires the **PAM**
reward-dopamine signal, losing fires **PPL1** (pain); both rewrite the
KC->action weights (a three-factor rule: pre KC activity x dopamine). Chips are
the fly's life -- at $0 it dies and a fresh fly takes its seat.

This is a fast *behavioral* model of that circuit so a whole table of flies can
play thousands of hands; it shares the mushroom-body principle with FlyGambler,
whose connectome-backed spiking version can validate individual decisions.
"""

from __future__ import annotations

import numpy as np

ACTIONS = ("fold", "call", "raise")


class FlyAgent:
    def __init__(
        self,
        *,
        n_kc: int = 800,
        kc_active: int = 40,
        lr: float = 0.12,
        temperature: float = 0.35,
        explore: float = 0.03,
        mood_gain: float = 0.15,
        mood_recovery: float = 0.97,
        tilt: float = 0.0,          # >0: losses -> looser/more aggressive (chasing)
        seed: int = 0,
        name: str = "fly",
    ):
        self.rng = np.random.default_rng(seed)
        self.n_kc = n_kc
        self.kc_active = kc_active
        self.lr = lr
        self.temperature = temperature
        self.explore = explore
        self.mood_gain = mood_gain
        self.mood_recovery = mood_recovery
        self.tilt = tilt
        self.name = name

        self.n_features = 8
        # fixed random projection features -> Kenyon cells (the sparse expansion)
        self.W_in = self.rng.normal(size=(n_kc, self.n_features))
        self.w = np.zeros((3, n_kc))     # learned KC -> action-value weights
        self.mood = 0.0                  # slow affect (for visualisation / optional tilt)

        # life stats (reset by the table when a fly is (re)seated)
        self.chips = 0.0
        self.alive = True
        self.hands = 0
        self.wins = 0
        self.born_hand = 0

    # ---- perception ----
    def _features(self, ctx: dict) -> np.ndarray:
        hs = ctx["hand_strength"]
        pot, to_call = ctx["pot"], ctx["to_call"]
        pot_odds = to_call / (pot + to_call + 1e-9)
        return np.array([
            1.0,                                   # bias
            hs, hs * hs, hs ** 3,                   # hand strength (nonlinear)
            pot_odds,                               # price to stay in
            ctx["n_active"] / ctx["n_seats"],       # how many opponents
            min(1.0, self.chips / ctx["start_chips"]),  # stack depth
            1.0 if hs > 0.5 else 0.0,               # coarse "is this a good hand"
        ])

    def _kc(self, feats: np.ndarray) -> np.ndarray:
        drive = self.W_in @ feats
        k = self.kc_active
        idx = np.argpartition(drive, -k)[-k:]      # k-winner-take-all sparse code
        h = np.zeros(self.n_kc)
        h[idx] = 1.0
        return h

    # ---- decide ----
    def act(self, ctx: dict):
        h = self._kc(self._features(ctx))
        Q = self.w @ h                              # value of fold/call/raise
        # tilt: a sad fly leans toward action (chasing) instead of folding
        if self.tilt:
            Q = Q + np.array([-1.0, 0.0, 1.0]) * self.tilt * max(0.0, -self.mood)
        z = np.clip(Q / self.temperature, -30.0, 30.0)
        z -= z.max()
        p = np.exp(z); p /= p.sum()
        p = self.explore / 3 + (1 - self.explore) * p
        a = int(self.rng.choice(3, p=p / p.sum()))
        return a, {"h": h, "Q": Q}

    # ---- learn ----
    def learn(self, ctx_learn: dict, action_idx: int, reward: float):
        """reward is the chip delta this hand in bet-units.

        Action-value (TD) learning: the error is measured against the value of
        the action actually taken, so each of fold/call/raise converges to its
        own expected payoff for this kind of hand -- which is what lets the fly
        learn to fold weak hands and bet strong ones.
        """
        self.hands += 1
        h = ctx_learn["h"]
        rpe = reward - float(ctx_learn["Q"][action_idx])
        # per-cell step scaled by the sparse-code size so the change in the
        # action value Q = w.h is ~lr*rpe (not kc_active*lr*rpe -> divergence)
        self.w[action_idx] += (self.lr / self.kc_active) * rpe * h
        self.mood = self.mood_recovery * self.mood + self.mood_gain * (reward if reward < 0 else 0.6 * reward)
        self.mood = float(np.clip(self.mood, -4.0, 3.0))
        dopamine = max(rpe, 0.0)                     # PAM (reward)
        pain = max(-rpe, 0.0)                        # PPL1 (punishment)
        return {"rpe": rpe, "dopamine": dopamine, "pain": pain, "mood": self.mood}
