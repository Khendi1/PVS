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

"""video_synth package.

Self-bootstraps ``sys.path`` so the package's internal *bare* imports
(e.g. ``from mixer import Mixer``) resolve no matter how the app is
launched. This removes the need to manually set
``PYTHONPATH=src:src/video_synth`` (which is also wrong on Windows, where
the path separator is ``;`` not ``:``).

After this runs, both of these directories are guaranteed to be on the
path whenever the package is imported:

* ``.../src/video_synth`` — for the bare internal imports
* ``.../src``            — for ``import video_synth`` / ``python -m video_synth``
"""

__version__ = "0.1.0"

import sys as _sys
from pathlib import Path as _Path

_PKG = _Path(__file__).resolve().parent  # .../src/video_synth
_SRC = _PKG.parent                       # .../src
for _p in (str(_PKG), str(_SRC)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

del _sys, _Path, _PKG, _SRC, _p
