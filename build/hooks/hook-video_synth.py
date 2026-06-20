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

# PyInstaller hook for video_synth package
# Dynamically collects all animation and effect submodules so none are missed
# even when new modules are added in the future.
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Pull every Python module under the video_synth package
hiddenimports = (
    collect_submodules('video_synth')
    + collect_submodules('video_synth.animations')
    + collect_submodules('video_synth.effects')
)

# Collect any non-Python data files that live inside the package directory
datas = collect_data_files('video_synth', includes=['**/*.yaml', '**/*.yml', '**/*.json'])
