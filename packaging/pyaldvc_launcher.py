"""Entry script for the frozen Windows bundle.

Kept outside the package on purpose: an ``al_dvc/__main__.py`` calls the CLI at
import time, so collecting it as an ordinary submodule could start a second
application. ``pyALDVC-console.exe --self-test [report]`` runs the
installation checks instead of opening the window.
"""

import multiprocessing
import sys


def _run() -> int:
    from al_dvc.gui.app import main

    return int(main(sys.argv) or 0)


if __name__ == "__main__":
    # Must precede anything else on Windows: a frozen build re-executes this
    # script in every child process.
    multiprocessing.freeze_support()
    sys.exit(_run())
