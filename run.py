#!/usr/bin/env python
# Video Synth — real-time collaborative visual art synthesizer.
# Copyright (C) 2026 Kyle Henderson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Zero-config launcher for the video synth.

Run this from the project root on any OS — no ``PYTHONPATH`` needed:

    python run.py                              # desktop GUI
    python run.py --headless --api             # headless API server
    python run.py --headless --api --no-virtualcam   # Docker / CI

Every CLI argument is forwarded verbatim to ``video_synth.__main__``,
so ``python run.py --help`` lists all available options.
"""
import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT / "src", _ROOT / "src" / "video_synth"):
    _entry = str(_p)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

# Execute video_synth/__main__.py as ``__main__`` so its existing
# ``if __name__ == "__main__"`` arg-parsing block runs and sys.argv (the
# forwarded CLI args) is honored. run_path executes the file exactly once
# (run_module on a package name re-runs the entry).
runpy.run_path(str(_ROOT / "src" / "video_synth" / "__main__.py"), run_name="__main__")
