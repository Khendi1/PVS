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
