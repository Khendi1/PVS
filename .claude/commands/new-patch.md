Write a new patch (preset) entry for the video synth's saved_values.yaml.

The user will describe a visual mood, scene, or concept. Your job is to:

1. **Read `write_patches.py`** to understand the helper format, LFO constants, and BASE_RESET defaults.
2. **Read `save/saved_values.yaml`** to see existing patch structure and avoid duplicate names.
3. **Read `video_synth/animations/enums.py`** to see available animations.
4. **Design the patch** with a clear artistic intent:
   - Choose src_1 and src_2 animations that contrast or complement each other
   - Pick a mixer blend mode (ALPHA_BLEND, LUMA_KEY, or CHROMA_KEY) that serves the composition
   - Set effect chain values (color shifts, warp, glitch, feedback) that reinforce the mood
   - Add 3–6 LFOs to key parameters to give the patch organic movement
   - Name the patch evocatively (e.g. "Bioluminescent Tide", "Neon Fungal Cathedral")

5. **Output a Python snippet** in the `write_patches.py` style (a dict ready to append to `entries`), plus the `lfo()` helper calls for each oscillator.

6. **Describe the visual result** in 2–3 sentences so the user can evaluate whether it matches their intent before committing the patch to the file.

7. **Ask if they want you to append it** to `write_patches.py` and run it, or manually paste the dict into `saved_values.yaml`.

LFO shape reference: NONE=0, SINE=1, SQUARE=2, TRIANGLE=3, SAWTOOTH=4, PERLIN=5
