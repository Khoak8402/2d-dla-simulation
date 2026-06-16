"""Config 3 - single seed, radial injection (R_inj = 180).

Walkers are re-injected on a circle of radius R_inj around the seed instead of
anywhere on the lattice, which gives a cleaner single radial aggregate.
"""

import dla_common as common

CONFIG_ID = 3

# --- Parameters here ---
PARAMS = dict(N=512, n_seeds=1, n_walkers=10000, max_steps=120000,
              reinject_timeout=1000, inj_mode="radial", inj_radius=180)


def main():
    return common.run_config(CONFIG_ID, PARAMS)


if __name__ == "__main__":
    main()
