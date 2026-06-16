"""Run every config in turn and produce all result images under results/.

Each config is seeded with the same fixed seed, so re-running this script
regenerates byte-identical PNGs.
"""

import test_config_1
import test_config_2
import test_config_3
import test_config_4


def main():
    for module in (test_config_1, test_config_2, test_config_3, test_config_4):
        module.main()


if __name__ == "__main__":
    main()
