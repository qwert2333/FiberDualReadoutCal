"""Resolution metrics shared by the CNN and non-CNN dual-readout baselines.

Kept free of torch so lightweight baselines can use the identical definitions.
"""

import numpy as np


def sigma68(values):
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.size < 2:
        return float("nan")
    n = max(2, int(round(0.68 * values.size)))
    return 0.5 * float(np.min(values[n - 1:] - values[:values.size - n + 1]))


def energy_metrics(values, true_energy):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return {"N": int(values.size), "mean_GeV": float("nan"),
                "mpv_GeV": float("nan"), "sigma68_GeV": float("nan"),
                "resolution_sigma68_over_mpv": float("nan"),
                "mean_relative_bias": float("nan")}
    lo, hi = float(np.min(values)), float(np.max(values))
    margin = 0.05 * max(hi - lo, 1.0e-6)
    hist, edges = np.histogram(values, bins=80, range=(lo - margin, hi + margin))
    peak = int(np.argmax(hist))
    mpv = 0.5 * float(edges[peak] + edges[peak + 1])
    width = sigma68(values)
    return {
        "N": int(values.size),
        "mean_GeV": float(np.mean(values)),
        "mpv_GeV": mpv,
        "sigma68_GeV": width,
        "resolution_sigma68_over_mpv": width / max(abs(mpv), 1.0e-9),
        "mean_relative_bias": (float(np.mean(values)) - true_energy) / true_energy,
    }
