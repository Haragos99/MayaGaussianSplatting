"""Train the MeSP scene with sensible defaults.

    python main.py                         # 7000 iterations at 480 px
    python main.py --iterations 1000       # any `gstrain train` option works
    python main.py --strategy mcmc         # 3DGS-MCMC with a splat cap
    python main.py verify out\\point_cloud.ply

Everything is forwarded to `gstrain.cli` (see `python -m gstrain.cli train -h`);
-s / -m given on the command line override the defaults below.
"""

import sys

from gstrain.cli import main

DEFAULT_SCENE = r"C:\Users\Geri\Documents\Projects\CG\MeSP"
# Not MeSP\out: that folder holds the old trainer's results.
DEFAULT_OUTPUT = DEFAULT_SCENE + r"\runs\latest"
COMMANDS = ("train", "verify", "stats")


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments[:1] and arguments[0] in COMMANDS:
        main(arguments)
    else:
        # argparse keeps the last value, so user flags override these defaults.
        main(["train", "-s", DEFAULT_SCENE, "-m", DEFAULT_OUTPUT, *arguments])
