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

from animations.plasma import Plasma
from animations.reaction_diffusion import ReactionDiffusion
from animations.metaballs import Metaballs
from animations.moire import Moire
from animations.strange_attractor import StrangeAttractor
from animations.physarum import Physarum
from animations.shaders import Shaders
from animations.shaders2 import Shaders2
from animations.dla import DLA
from animations.chladni import Chladni
from animations.voronoi import Voronoi
from animations.drift_field import DriftField
from animations.lenia import Lenia
from animations.fractal_zoom import FractalZoom
from animations.oscillator_grid import OscillatorGrid
from animations.harmonic_interference import HarmonicInterference
from animations.shaders3 import Shaders3

__all__ = [
    "Plasma",
    "ReactionDiffusion",
    "Metaballs",
    "Moire",
    "StrangeAttractor",
    "Physarum",
    "Shaders",
    "Shaders2",
    "Shaders3",
    "DLA",
    "Chladni",
    "Voronoi",
    "DriftField",
    "Lenia",
    "FractalZoom",
    "OscillatorGrid",
    "HarmonicInterference",
]
