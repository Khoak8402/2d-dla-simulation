"""Config 1 - single seed, random injection.

Follows the sample test template: set parameters, run the simulation, then
visualize and capture the three standard plots. `seed_all()` (called inside
run_config) is what keeps the result images identical on every run.
"""

import dla_common as common

CONFIG_ID = 1

# --- Parameters here ---
PARAMS = dict(N=512, n_seeds=1, n_walkers=10000, max_steps=80000,
              reinject_timeout=1000, inj_mode="random", inj_radius=None)


def main():
    # Seeds both RNGs, runs dla.simulation(**PARAMS), and captures:
    #   results/config1_growth_stages.png   (display_aggregate_grow)
    #   results/config1_mass_radius.png      (display_mass_radius_plot)
    #   results/config1_progress.png         (display_grow_progress_plot)
    return common.run_config(CONFIG_ID, PARAMS)


if __name__ == "__main__":
    main()
