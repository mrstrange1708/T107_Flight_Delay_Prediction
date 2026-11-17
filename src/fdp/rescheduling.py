"""Phase 2 · offline "what-if" rescheduling analysis.
Owner: Shaik Junaid Sami (Sami).

The recommendation rule: for a flight the model flags at 05:00, ADD A BUFFER of B minutes to its
scheduled arrival (pad the block time). The question is whether spending buffer minutes on the
model's flags avoids more late arrivals than spending the SAME minutes another way.

Explicit assumptions (the analysis is only as good as these):
  A1  Padding changes the published arrival time only; the plane lands when it would have landed.
      So a padded flight is late if  arr_delay - B >= 15.
  A2  No knock-on effects: padding one flight does not change the aircraft's next flight, crew
      legality, gate plans or passenger connections.
  A3  Cost = buffer minutes added (aircraft and crew time). Every policy gets the same budget.
      A buffer minute is "used" if the flight was at least that late, and idle otherwise.
  A4  Actual arrival delays of the test months are used ONLY to score the policy offline. The
      flags themselves use 05:00 information only, exactly as in prediction.
  A5  Cancelled and diverted flights have no arrival time and are left out.
"""
import numpy as np
import pandas as pd

LATE = 15


def avoided(arr_delay, padded_mask, buffer_min):
    """Late arrivals turned on-time by padding `padded_mask` flights by `buffer_min` minutes (A1)."""
    a = np.asarray(arr_delay, float)
    b = np.where(padded_mask, buffer_min, 0.0)
    return int(((a >= LATE) & (a - b < LATE)).sum())


def top_k_mask(score, k):
    m = np.zeros(len(score), bool)
    if k > 0:
        m[np.argsort(-np.asarray(score))[:k]] = True
    return m


def absorbed_minutes(arr_delay, padded_mask, buffer_min):
    """Minutes of late arrival (vs the published schedule) that the buffer soaks up (A1).
    A flight that arrives early or on time absorbs nothing: its buffer minutes are idle."""
    a = np.clip(np.asarray(arr_delay, float), 0, None)
    return float(np.minimum(a, buffer_min)[padded_mask].sum())


def compare_policies(arr_delay, scores: dict, k: int, buffer_min: float, seed: int = 0, n_random: int = 200):
    """The same budget (k flights x buffer_min minutes) spent by different policies.

    scores: {"policy name": risk score}; each policy pads its own top-k flights.
    A "random flights" policy (mean of n_random draws) is the floor every score must beat.
    (Spreading the budget as fractions of a minute over every flight is not a schedule anyone
    can publish, so it is not a policy here.)"""
    a = np.asarray(arr_delay, float)
    n, budget = len(a), k * buffer_min
    late = int((a >= LATE).sum())
    rows = []
    for name, s in scores.items():
        m = top_k_mask(s, k)
        rows.append((name, avoided(a, m, buffer_min), absorbed_minutes(a, m, buffer_min)))
    rng = np.random.default_rng(seed)
    draws = [top_k_mask(rng.random(n), k) for _ in range(n_random)]
    rows.append(("random flights", float(np.mean([avoided(a, m, buffer_min) for m in draws])),
                 float(np.mean([absorbed_minutes(a, m, buffer_min) for m in draws]))))
    out = pd.DataFrame(rows, columns=["policy", "late arrivals avoided", "delay minutes absorbed"])
    out["avoided per 1,000 flights"] = 1000 * out["late arrivals avoided"] / n
    out["share of all late arrivals avoided"] = out["late arrivals avoided"] / late
    out["share of buffer minutes used"] = out["delay minutes absorbed"] / budget
    return out


def buffer_sweep(arr_delay, score, k, buffers=(5, 10, 15, 20, 30, 45, 60)):
    """For the model's k flags: what each buffer size buys, and what it wastes."""
    a = np.asarray(arr_delay, float)
    m = top_k_mask(score, k)
    rows = []
    for b in buffers:
        av = avoided(a, m, b)
        rows.append({"buffer (min)": b, "late arrivals avoided": av,
                     "still late after the buffer": int(((a >= LATE) & m & (a - b >= LATE)).sum()),
                     "padded but on time anyway": int((m & (a < LATE)).sum()),
                     "avoided per 1,000 buffer minutes": 1000 * av / (k * b),
                     "share of buffer minutes used": absorbed_minutes(a, m, b) / (k * b)})
    return pd.DataFrame(rows)
