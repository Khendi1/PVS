Scaffold a new effect module for the video synth.

The user will describe the visual effect they want. Your job is to:

1. **Ask for the effect name** if not provided (PascalCase, e.g. `ChromaShift`).

2. **Generate the full effect file** at `video_synth/effects/<snake_case_name>.py` following the canonical pattern from `video_synth/effects/color.py`:
   - Import `EffectBase` from `effects.base`, `Widget` from `common`
   - Constructor takes `(self, params, group)`
   - Set `subgroup = self.__class__.__name__`
   - Register all params via `params.new(name, min=, max=, default=, subgroup=subgroup, group=group)`
   - The primary method is named `do_<effect>(self, frame: np.ndarray) -> np.ndarray`
   - When the effect is disabled (all params at zero/default), return the frame unmodified immediately — no wasted compute
   - Use vectorized NumPy / OpenCV; avoid Python loops over pixels
   - Include a short docstring explaining what the effect does visually

3. **List the registration steps** the user must take manually:
   - Import and instantiate in `effects_manager.py` (add to the `EffectManager.__init__` and call `do_xxx` in the effect chain)
   - Add a PARAMETERS.md section for each param

4. **Suggest which existing effects** it would pair well with and what order in the chain makes sense (e.g., color shifts before glitch, warp before feedback).

5. **Suggest 1–2 LFO targets** that would give interesting animation.

Do not create any extra files beyond the effect module itself unless asked.
