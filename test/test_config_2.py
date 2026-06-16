"""Config 2 - multiple seeds (12), random injection.

With several seeds the cluster grows from many points at once. D_f is not
meaningful here (it is measured about a single center), so the mass-radius
plot may report too little data - that is expected for this config.
"""

import dla_common as common

CONFIG_ID = 2

# --- Parameters here ---
PARAMS = dict(N=512, n_seeds=12, n_walkers=15000, max_steps=80000,
              reinject_timeout=1000, inj_mode="random", inj_radius=None)


def main():
    return common.run_config(CONFIG_ID, PARAMS)


if __name__ == "__main__":
    main()
