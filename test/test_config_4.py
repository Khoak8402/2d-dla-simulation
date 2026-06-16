"""Config 4 - single seed, radial injection, high particle count.

Same radial setup as Config 3 but with many more walkers, producing a larger,
denser aggregate that captures incoming walkers far more efficiently.
"""

import dla_common as common

CONFIG_ID = 4

# --- Parameters here ---
PARAMS = dict(N=512, n_seeds=1, n_walkers=25000, max_steps=200000,
              reinject_timeout=1000, inj_mode="radial", inj_radius=180)


def main():
    return common.run_config(CONFIG_ID, PARAMS)


if __name__ == "__main__":
    main()
