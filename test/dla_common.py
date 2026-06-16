"""Shared helpers for the DLA config tests.

Two responsibilities only:
  1. Make every run reproducible, so the result images never change.
  2. Capture the three standard `visualization.display_*` plots to PNG files.

Reproducibility note
--------------------
The walkers move with random numbers, so identical images require an identical
random stream every run. There are TWO separate RNGs to seed:
  * NumPy's RNG  -> used by the plain-Python parts of dla_simulation.
  * Numba's RNG  -> used inside every @jit(nopython=True) function; it is a
                    DIFFERENT generator and is only seeded from inside njit code.
`seed_all()` seeds both. Seed once, right before `dla.simulation(...)`.
"""

import os
import sys
import time

import numpy as np
from numba import njit

# --- make the repo root importable (so `import dla_simulation` works) ---------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Headless, deterministic rendering backend. MUST be selected before pyplot and
# before importing `visualization` (which imports pyplot).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dla_simulation as dla          # noqa: E402  (import after sys.path tweak)
import visualization as vis           # noqa: E402

RESULTS_DIR = os.path.join(_REPO_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Fixed seed -> identical random walks -> identical aggregates -> identical PNGs.
SEED = 42


@njit
def _seed_numba(s):
    # Seeds Numba's own RNG. Without this the @jit walker code reseeds itself
    # nondeterministically and the result images change run-to-run.
    np.random.seed(s)


def seed_all(seed=SEED):
    """Seed BOTH RNGs. Call once, immediately before dla.simulation(...)."""
    np.random.seed(seed)
    _seed_numba(seed)


def capture(display_fn, results, path):
    """Run a `visualization.display_*` function but save its figure to `path`
    instead of opening a window.

    The display functions end with `plt.show()`, which would block / pop a
    window. We temporarily replace `plt.show` with a no-op, let the function
    build its figure, then save whatever figure it created. PNGs written by the
    Agg backend contain no timestamp, so re-running produces byte-identical
    files as long as the seed is fixed.
    """
    real_show = plt.show
    plt.show = lambda *a, **k: None          # swallow the interactive show()
    before = set(plt.get_fignums())
    try:
        display_fn(results)
        new = set(plt.get_fignums()) - before
        if not new:
            # e.g. display_mass_radius_plot bails out when there is too little data
            print(f"  (no figure from {display_fn.__name__}; skipped "
                  f"{os.path.basename(path)})")
            return None
        fig = plt.figure(max(new))
        fig.savefig(path, dpi=130, bbox_inches="tight")
        return path
    finally:
        plt.close("all")
        plt.show = real_show


def run_config(config_id, params, seed=SEED):
    """Seed -> simulate -> capture the three plots -> print a one-line summary.

    Shared by every test_config_*.py so they stay short and identical in shape.
    Returns the `results` dict from the simulation.
    """
    seed_all(seed)
    t0 = time.perf_counter()
    results = dla.simulation(**params)
    elapsed = time.perf_counter() - t0

    tag = f"config{config_id}"
    capture(vis.display_aggregate_grow, results,
            os.path.join(RESULTS_DIR, f"{tag}_growth_stages.png"))
    capture(vis.display_mass_radius_plot, results,
            os.path.join(RESULTS_DIR, f"{tag}_mass_radius.png"))
    capture(vis.display_grow_progress_plot, results,
            os.path.join(RESULTS_DIR, f"{tag}_progress.png"))

    df = results["fractal_dimension"]
    df_str = f"{df:.3f}" if df == df else "n/a"       # df != df  <=>  df is NaN
    print(f"[Config {config_id}] "
          f"particles={int(results['n_particles'])}  "
          f"clusters={int(results['n_aggregate'])}  "
          f"D_f={df_str}  "
          f"time={elapsed:.2f}s")
    return results
