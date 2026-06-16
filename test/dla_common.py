import os
import sys
import time
import json

import numpy as np
from numba import njit

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dla_simulation as dla

RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DF_THEORY = 1.71


@njit
def _seed_numba(s):
    np.random.seed(s)


def seed_all(seed):
    np.random.seed(seed)
    _seed_numba(seed)


def _iters_to_fraction(time_history, growth_history, target_count, fraction):
    if len(growth_history) == 0:
        return None
    threshold = fraction * target_count
    growth = np.asarray(growth_history)
    times = np.asarray(time_history)
    idx = np.argmax(growth >= threshold)
    if growth[idx] < threshold:
        return None
    if idx == 0:
        return float(times[0])
    g0, g1 = growth[idx - 1], growth[idx]
    t0, t1 = times[idx - 1], times[idx]
    if g1 == g0:
        return float(t1)
    return float(t0 + (t1 - t0) * (threshold - g0) / (g1 - g0))


def run_config(name, params, seed=42, measure_df=True):
    seed_all(seed)
    t0 = time.perf_counter()
    results = dla.simulation(**params)
    elapsed = time.perf_counter() - t0

    n_walkers = params["n_walkers"]
    n_seeds = params["n_seeds"]
    n_stuck = int(results["n_particles"])
    aggregated = max(n_stuck - n_seeds, 0)

    time_history = results["time-history"]
    growth_history = results["growth-history"]
    iters_sampled = int(time_history[-1]) if len(time_history) else 0

    metrics = {
        "name": name,
        "seed": seed,
        "N": params["N"],
        "n_seeds": n_seeds,
        "n_walkers": n_walkers,
        "injection": params["inj_mode"],
        "inj_radius": params.get("inj_radius"),
        "n_stuck": n_stuck,
        "aggregated": aggregated,
        "completion_rate": aggregated / n_walkers if n_walkers else 0.0,
        "n_aggregate": int(results["n_aggregate"]),
        "wall_time_s": elapsed,
        "iters_sampled": iters_sampled,
        "throughput_pps": n_stuck / elapsed if elapsed else 0.0,
        "efficiency_ppi": n_stuck / iters_sampled if iters_sampled else 0.0,
        "iters_to_25pct": _iters_to_fraction(time_history, growth_history, n_walkers, 0.25),
        "iters_to_50pct": _iters_to_fraction(time_history, growth_history, n_walkers, 0.50),
        "iters_to_75pct": _iters_to_fraction(time_history, growth_history, n_walkers, 0.75),
        "iters_to_90pct": _iters_to_fraction(time_history, growth_history, n_walkers, 0.90),
    }

    if measure_df:
        df = results["fractal_dimension"]
        metrics["fractal_dimension"] = float(df) if df == df else None
    else:
        metrics["fractal_dimension"] = None

    return {"results": results, "metrics": metrics, "elapsed": elapsed}


def save_structure(results, path, suptitle):
    snapshots = results["snapshots"]
    glued_counts = results["glued_counts"]
    total = results["n_particles"]

    targets = [0.10 * total, 0.50 * total, total]
    indices = [int(np.argmin(np.abs(glued_counts - t))) for t in targets]
    titles = ["~10% progress", "~50% progress", "100% (final)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=0.97)
    for ax, idx, title in zip(axes, indices, titles):
        binary_map = np.where(snapshots[idx] == 2, 1, 0)
        ax.imshow(binary_map, cmap="magma", interpolation="nearest")
        ax.set_title(f"{title}\n({int(glued_counts[idx])} particles)", fontsize=12)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_mass_radius(results, path, title):
    radii = results["radii"]
    masses = results["masses"]
    max_radius = np.max(radii)
    valid = (masses > 10) & (radii < max_radius * 0.8)
    if np.sum(valid) <= 5:
        return None

    log_r = np.log(radii[valid])
    log_m = np.log(masses[valid])
    df, c = np.polyfit(log_r, log_m, 1)

    fig = plt.figure(figsize=(8, 6))
    plt.plot(log_r, log_m, "o", color="royalblue", label="Measured data", alpha=0.7)
    plt.plot(log_r, df * log_r + c, "r-", linewidth=2.5,
             label=f"Simulation fit ($D_f$ = {df:.3f})")
    plt.plot(log_r, DF_THEORY * log_r + c, "g--", linewidth=2.5,
             label=f"Theory ($D_f$ = {DF_THEORY})")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("$\\log(R)$", fontsize=12)
    plt.ylabel("$\\log(M)$", fontsize=12)
    plt.legend(fontsize=11, loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return float(df)


def save_growth(results, path, title):
    time_history = results["time-history"]
    growth_history = results["growth-history"]
    total = results["n_particles"]

    fig = plt.figure(figsize=(8, 5))
    plt.plot(time_history, growth_history, color="royalblue", linewidth=2.5)
    plt.axhline(y=total, color="gray", linestyle=":", alpha=0.8,
                label=f"Final ({int(total)} particles)")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Glued particles", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_metrics_json(metrics, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


def print_metrics(metrics):
    m = metrics
    print(f"--- {m['name']} ---")
    print(f"  grid N={m['N']}  seeds={m['n_seeds']}  walkers={m['n_walkers']}  "
          f"injection={m['injection']}"
          + (f" (R_inj={m['inj_radius']})" if m['injection'] == 'radial' else ""))
    print("  [completion] "
          f"stuck={m['n_stuck']}  aggregated={m['aggregated']}/{m['n_walkers']}  "
          f"rate={m['completion_rate']*100:.1f}%  clusters={m['n_aggregate']}")
    print("  [speed]      "
          f"wall={m['wall_time_s']:.2f}s  iters~={m['iters_sampled']}  "
          f"throughput={m['throughput_pps']:.0f} part/s  "
          f"efficiency={m['efficiency_ppi']:.3f} part/iter")
    print("  [growth]     iters to "
          f"25%={_fmt(m['iters_to_25pct'])}  50%={_fmt(m['iters_to_50pct'])}  "
          f"75%={_fmt(m['iters_to_75pct'])}  90%={_fmt(m['iters_to_90pct'])}")
    if m["fractal_dimension"] is not None:
        print(f"  [fractal]    D_f={m['fractal_dimension']:.3f} "
              f"(theory {DF_THEORY})")
    else:
        print("  [fractal]    D_f not measured for this configuration")


def _fmt(v):
    return "n/a" if v is None else f"{v:.0f}"


CONFIGS = {
    1: {
        "name": "Config 1: single seed, random injection",
        "measure_df": True,
        "params": dict(N=512, n_seeds=1, n_walkers=10000, max_steps=80000,
                       reinject_timeout=1000, inj_mode="random", inj_radius=None),
    },
    2: {
        "name": "Config 2: multi-seed (12), random injection",
        "measure_df": False,
        "params": dict(N=512, n_seeds=12, n_walkers=15000, max_steps=80000,
                       reinject_timeout=1000, inj_mode="random", inj_radius=None),
    },
    3: {
        "name": "Config 3: single seed, radial injection (R_inj=180)",
        "measure_df": True,
        "params": dict(N=512, n_seeds=1, n_walkers=10000, max_steps=120000,
                       reinject_timeout=1000, inj_mode="radial", inj_radius=180),
    },
    4: {
        "name": "Config 4: single seed, radial injection, high density",
        "measure_df": True,
        "params": dict(N=512, n_seeds=1, n_walkers=25000, max_steps=200000,
                       reinject_timeout=1000, inj_mode="radial", inj_radius=180),
    },
}


def run_and_save(cfg_id, seed=42):
    cfg = CONFIGS[cfg_id]
    bundle = run_config(cfg["name"], cfg["params"], seed=seed,
                        measure_df=cfg["measure_df"])
    results, metrics = bundle["results"], bundle["metrics"]

    save_structure(results, os.path.join(RESULTS_DIR, f"config{cfg_id}_structure.png"),
                   cfg["name"])
    save_growth(results, os.path.join(RESULTS_DIR, f"config{cfg_id}_growth.png"),
                f"Growth progress - {cfg['name']}")
    if cfg["measure_df"]:
        df = save_mass_radius(
            results, os.path.join(RESULTS_DIR, f"config{cfg_id}_mass_radius.png"),
            f"Mass-radius fit - {cfg['name']}")
        if df is not None:
            metrics["fractal_dimension"] = df

    save_metrics_json(metrics, os.path.join(RESULTS_DIR, f"config{cfg_id}_metrics.json"))
    print_metrics(metrics)
    return bundle
