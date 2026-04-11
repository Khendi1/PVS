Scaffold a new animation module for the video synth.

The user will describe the animation they want to create. Your job is to:

1. **Ask for the animation name** if not provided (PascalCase class name, e.g. `LorenzCurve`).

2. **Generate the full animation file** at `video_synth/animations/<snake_case_name>.py` following the canonical pattern from `video_synth/animations/metaballs.py`:
   - Import `Animation` from `animations.base`, `Widget` from `common`, colormaps if needed
   - Constructor takes `(self, params, width, height, group=None)`
   - Call `super().__init__(params, width, height, group=group)`
   - Set `subgroup = self.__class__.__name__`
   - Register ALL tunable values via `params.new(name, min=, max=, default=, subgroup=subgroup, group=group)`
   - Cache expensive computed data (meshgrids, precomputed arrays) as instance vars
   - Implement `get_frame(self, frame: np.ndarray) -> np.ndarray`
   - Use vectorized NumPy — no Python loops over pixels
   - Include a frame-rate budget comment (target ≤ 10 ms render time)

3. **List the registration steps** the user must take manually:
   - Add the class to `animations/enums.py` (AnimationType enum)
   - Import and instantiate in `effects_manager.py`
   - Add a PARAMETERS.md entry for each param

4. **Suggest 2–3 interesting LFO targets** from the new params that would make the animation feel alive in a patch.

5. **Optionally write a sample patch entry** using the `write_patches.py` helper style if the user asks.

Do not create any extra files beyond the animation module itself unless asked.
