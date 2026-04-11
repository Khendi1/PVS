import yaml

def lfo(name, shape=1, freq=0.1, amp=1.0, phase=0.0, seed=0, cutoff_min=None, cutoff_max=None):
    d = {
        f'{name}_shape': shape,
        f'{name}_frequency': freq,
        f'{name}_amplitude': amp,
        f'{name}_phase': phase,
        f'{name}_seed': seed,
        f'{name}_noise_octaves': 6,
        f'{name}_noise_persistence': 0.5,
        f'{name}_noise_lacunarity': 2.0,
        f'{name}_noise_repeat': 100,
        f'{name}_noise_base': seed * 100,
    }
    if cutoff_min is not None: d[f'{name}_cutoff_min'] = cutoff_min
    if cutoff_max is not None: d[f'{name}_cutoff_max'] = cutoff_max
    return d

SINE=1; SQUARE=2; TRIANGLE=3; SAWTOOTH=4; PERLIN=5; LFO_NONE=0

BASE_RESET = {
    'blend_mode': 0, 'alpha_blend': 0.5, 'swap_sources': 0,
    'alpha': 0.0, 'temporal_filter': 0.0, 'feedback_luma_threshold': 0,
    'luma_mode': 1, 'frame_skip': 0, 'buffer_frame_select': -1,
    'buffer_frame_blend': 0.0, 'prev_frame_scale': 100, 'buffer_size': 0,
    'fb_paint_drift_x': 0.0, 'fb_paint_drift_y': 0.0,
    'fb_paint_rotation': 0.0, 'fb_paint_zoom': 1.0,
    'hue_shift': 0, 'sat_shift': 0, 'val_shift': 0,
    'contrast': 1.0, 'brightness': 0, 'gamma': 1.0,
    'color_cycle_speed': 0.0, 'color_cycle_bands': 3,
    'false_color_strength': 0.0, 'false_color_map': 14,
    'invert_strength': 0.0, 'hue_scatter': 0.0, 'duotone_strength': 0.0,
    'solarize_threshold': 0.0, 'hue_invert_strength': 0.0,
    'highlight_compression': 0.0, 'color_temp': 0.0,
    'chroma_ab_x': 0, 'chroma_ab_y': 0, 'color_bitcrush': 8,
    'ch_r': 1.0, 'ch_g': 1.0, 'ch_b': 1.0,
    'ch_mix_rr': 1.0, 'ch_mix_rg': 0.0, 'ch_mix_rb': 0.0,
    'ch_mix_gr': 0.0, 'ch_mix_gg': 1.0, 'ch_mix_gb': 0.0,
    'ch_mix_br': 0.0, 'ch_mix_bg': 0.0, 'ch_mix_bb': 1.0,
    'sat_curve_shadows': 1.0, 'sat_curve_mids': 1.0, 'sat_curve_highlights': 1.0,
    'warp_type': 0,
    'enable_pixel_shift': 0, 'enable_color_split': 0,
    'enable_block_corruption': 0, 'enable_random_rectangles': 0,
    'enable_horizontal_scroll_freeze': 0, 'num_glitches': 0,
    'enable_slitscan': 0, 'enable_echo': 0,
    'shape_type': 0,
    'blur_type': 0, 'blur_kernel_size': 1, 'noise_type': 0,
    'x_shift': 0, 'y_shift': 0, 'zoom': 1.0, 'r_shift': 0.0,
    'prev_x_shift': 0, 'prev_y_shift': 0, 'prev_zoom': 1.0, 'prev_r_shift': 0.0,
    'reflection_mode': 0, 'reflector_segments': 0,
    'reflector_z': 1.0, 'reflector_r': 0.0,
    'erosion_strength': 0.0,
    'lissajous_brightness': 0,
    **lfo('plasma_speed', LFO_NONE, 0.5, 1.0),
    **lfo('plasma_distance', LFO_NONE, 0.5, 1.0),
    **lfo('plasma_color_speed', LFO_NONE, 0.5, 1.0),
    **lfo('plasma_flow_speed', LFO_NONE, 0.5, 1.0),
    **lfo('src_1_animations_moire_rot1', LFO_NONE, 0.1, 180.0),
    **lfo('src_1_animations_moire_rot2', LFO_NONE, 0.07, 180.0),
    **lfo('src_2_animations_moire_rot1', LFO_NONE, 0.1, 180.0),
    **lfo('src_2_animations_moire_rot2', LFO_NONE, 0.07, 180.0),
}

# -----------------------------------------------------------------------
# PATCH 1: Slime Cathedral
# Physarum agents vs tunnel shader. Heavy feedback with slow rotation
# creates cathedral-like chambers of flowing organic light.
# -----------------------------------------------------------------------
patch1 = {**BASE_RESET,
    '_name': 'Slime Cathedral',
    'source_1': 'PHYSARUM',
    'source_2': 'SHADERS_2',
    'alpha_blend': 0.25,
    'phys_num_agents': 8000,
    'phys_sensor_distance': 14,
    'phys_sensor_angle_spacing': 0.45,
    'phys_turn_angle': 0.62,
    'phys_step_distance': 2,
    'phys_decay_factor': 0.04,
    'phys_diffuse_factor': 0.8,
    'phys_deposit_amount': 2.5,
    'phys_grid_res_scale': 0.5,
    'phys_wrap_around': 1,
    'phys_trail_r': 0,   'phys_trail_g': 160, 'phys_trail_b': 255,
    'phys_agent_r': 255, 'phys_agent_g': 220, 'phys_agent_b': 0,
    'phys_agent_size': 1,
    's2_type': 4,  # TUNNEL
    's2_zoom': 1.8, 's2_speed': 0.3, 's2_brightness': 0.5,
    's2_distortion': 0.6, 's2_iterations': 8.0, 's2_color_shift': 1.4,
    'alpha': 0.86,
    'fb_paint_rotation': -0.4,
    'fb_paint_zoom': 1.002,
    'fb_paint_drift_y': 0.1,
    'feedback_luma_threshold': 20,
    'false_color_strength': 0.55,
    'false_color_map': 15,  # PLASMA
    'color_cycle_speed': 0.08,
    'sat_curve_shadows': 1.5,
    'gamma': 0.85,
}

# -----------------------------------------------------------------------
# PATCH 2: Lorenz Storm
# Lorenz attractor orbiting through a turbulent convective field.
# Slow morph between attractor configurations. Long orbital trails.
# -----------------------------------------------------------------------
patch2 = {**BASE_RESET,
    '_name': 'Lorenz Storm',
    'source_1': 'STRANGE_ATTRACTOR',
    'source_2': 'DRIFT_FIELD',
    'alpha_blend': 0.35,
    'attractor_type': 0,  # LORENZ
    'attractor_dt': 0.013,
    'attractor_num_steps': 25,
    'attractor_scale': 9.0,
    'attractor_line_width': 1,
    'attractor_fade': 0.97,
    'attractor_r': 80, 'attractor_g': 140, 'attractor_b': 255,
    'attractor_morph_speed': 0.015,
    'lorenz_sigma': 10.0, 'lorenz_rho': 28.0, 'lorenz_beta': 2.667,
    'drift_speed': 0.08,
    'drift_complexity': 5,
    'drift_scale': 4.0,
    'drift_viscosity': 0.998,
    'drift_injection': 0.015,
    'drift_colormap': 18,  # TWILIGHT
    'drift_color_speed': 0.4,
    'warp_type': 8,  # CONVECTION
    'conv_rise_speed': 1.5,
    'conv_diffusion': 0.4,
    'conv_turbulence': 0.5,
    'conv_decay': 0.97,
    'alpha': 0.7,
    'fb_paint_drift_x': 0.2,
    'fb_paint_drift_y': -0.1,
    'fb_paint_rotation': 0.3,
    'color_cycle_speed': 0.25,
    'color_cycle_bands': 2,
    'gamma': 1.1, 'contrast': 1.2,
}

# -----------------------------------------------------------------------
# PATCH 3: Spinning Mandala
# Two moire layers (spiral vs radial) with LFO counter-rotation.
# Kaleidoscope reflection turns the interference pattern into a mandala.
# -----------------------------------------------------------------------
patch3 = {**BASE_RESET,
    '_name': 'Spinning Mandala',
    'source_1': 'MOIRE',
    'source_2': 'MOIRE',
    'alpha_blend': 0.5,
    'moire_type_1': 3,    # SPIRAL
    'spatial_freq_1': 8.0,
    'angle_1': 0.0, 'zoom_1': 0.8,
    'moire_blend': 0,     # MULTIPLY
    'moire_type_2': 1,    # RADIAL
    'spatial_freq_2': 6.0,
    'angle_2': 0.0, 'zoom_2': 1.0,
    **lfo('src_1_animations_moire_rot1', SINE,     0.05,  180.0,   0.0, 0, 0, 360),
    **lfo('src_1_animations_moire_rot2', SINE,     0.031, 180.0,  90.0, 7, 0, 360),
    **lfo('src_2_animations_moire_rot1', SINE,     0.04,  180.0, 180.0, 3, 0, 360),
    **lfo('src_2_animations_moire_rot2', TRIANGLE, 0.02,  180.0, 270.0,11, 0, 360),
    'reflection_mode': 1,
    'reflector_segments': 6,
    'reflector_z': 1.0,
    'reflector_r': 15.0,
    'color_cycle_speed': 0.6,
    'color_cycle_bands': 4,
    'sat_curve_shadows': 1.8,
    'contrast': 1.3,
    'alpha': 0.3,
    'fb_paint_rotation': 0.6,
}

# -----------------------------------------------------------------------
# PATCH 4: Plasma Glitch Storm
# All plasma LFOs on Perlin noise - every param drifts unpredictably.
# Clifford attractor bleeds in. Chromatic aberration and slit-scan
# push it into brutal digital artifact territory.
# -----------------------------------------------------------------------
patch4 = {**BASE_RESET,
    '_name': 'Plasma Glitch Storm',
    'source_1': 'PLASMA',
    'source_2': 'STRANGE_ATTRACTOR',
    'alpha_blend': 0.6,
    'plasma_speed': 3.0,
    'plasma_distance': 2.0,
    'plasma_color_speed': 2.5,
    'plasma_flow_speed': 1.5,
    **lfo('plasma_speed',       PERLIN, 0.15, 4.5, 0.0,   1, 0.1, 10.0),
    **lfo('plasma_distance',    PERLIN, 0.09, 4.0, 120.0, 22, 0.1,  9.0),
    **lfo('plasma_color_speed', PERLIN, 0.2,  4.0, 240.0, 55, 0.1,  9.0),
    **lfo('plasma_flow_speed',  SINE,   0.12, 3.0, 60.0,   8, 0.5,  8.0),
    'attractor_type': 1,  # CLIFFORD
    'attractor_dt': 0.01, 'attractor_num_steps': 15,
    'attractor_scale': 6.0, 'attractor_fade': 0.93,
    'attractor_r': 255, 'attractor_g': 80, 'attractor_b': 120,
    'clifford_a': -1.4, 'clifford_b': 1.6, 'clifford_c': 1.0, 'clifford_d': 0.7,
    'enable_pixel_shift': 1,
    'enable_color_split': 1,
    'glitch_intensity_max': 35,
    'glitch_duration_frames': 45,
    'glitch_band_div': 8,
    'num_glitches': 15,
    'glitch_size': 20,
    'enable_slitscan': 1,
    'slitscan_direction': 0,
    'slitscan_slice_width': 3,
    'slitscan_time_offset': 15,
    'slitscan_speed': 2.0,
    'slitscan_buffer_size': 30,
    'chroma_ab_x': 8, 'chroma_ab_y': 4,
    'alpha': 0.55,
    'temporal_filter': 0.15,
    'hue_scatter': 0.08,
    'color_bitcrush': 6,
}

# -----------------------------------------------------------------------
# PATCH 5: Reaction Bloom
# Reaction diffusion (coral maze) luma-keyed over Lenia (soft life forms).
# RD warp distorts both. False color maps cell density to INFERNO.
# -----------------------------------------------------------------------
patch5 = {**BASE_RESET,
    '_name': 'Reaction Bloom',
    'source_1': 'REACTION_DIFFUSION',
    'source_2': 'LENIA',
    'blend_mode': 1,   # LUMA_KEY
    'luma_threshold': 90,
    'luma_selection': 1,
    'luma_blur': 5,
    'da': 1.0, 'db': 0.5,
    'feed': 0.055, 'kill': 0.062,
    'iterations_per_frame': 60,
    'lenia_dt': 0.12, 'lenia_mu': 0.14, 'lenia_sigma': 0.016,
    'lenia_radius': 15, 'lenia_colormap': 15, 'lenia_seed_density': 0.12,
    'warp_type': 9,  # RD_WARP
    'rd_warp_strength': 18.0,
    'rd_warp_feed': 0.055, 'rd_warp_kill': 0.062, 'rd_warp_speed': 0.8,
    'alpha': 0.4,
    'fb_paint_zoom': 1.001,
    'false_color_strength': 0.75,
    'false_color_map': 14,   # INFERNO
    'gamma': 0.9,
    'sat_curve_highlights': 1.4,
    'highlight_compression': 0.2,
}

# -----------------------------------------------------------------------
# PATCH 6: Aizawa Orbit
# Aizawa attractor (spiraling torus-like) over a twilight drift field.
# Very long fade creates layered orbital ghosting. Perlin warp adds
# a gentle breathing motion. Slow morph drifts through parameter space.
# -----------------------------------------------------------------------
patch6 = {**BASE_RESET,
    '_name': 'Aizawa Orbit',
    'source_1': 'STRANGE_ATTRACTOR',
    'source_2': 'DRIFT_FIELD',
    'alpha_blend': 0.45,
    'attractor_type': 3,  # AIZAWA
    'attractor_dt': 0.008, 'attractor_num_steps': 30,
    'attractor_scale': 12.0, 'attractor_line_width': 1,
    'attractor_fade': 0.99,
    'attractor_r': 200, 'attractor_g': 100, 'attractor_b': 255,
    'attractor_morph_speed': 0.008,
    'aizawa_a': 0.95, 'aizawa_b': 0.7,  'aizawa_c': 0.6,
    'aizawa_d': 3.5,  'aizawa_e': 0.25, 'aizawa_f': 0.1,
    'drift_speed': 0.05,
    'drift_complexity': 6,
    'drift_scale': 5.0,
    'drift_viscosity': 0.999,
    'drift_injection': 0.01,
    'drift_colormap': 19,   # TWILIGHT_SHIFTED
    'drift_color_speed': 0.15,
    'warp_type': 4,  # PERLIN
    'warp_speed': 8,
    'x_size': 25.0, 'y_size': 20.0,
    'x_speed': 0.5, 'y_speed': 0.5,
    'alpha': 0.91,
    'fb_paint_rotation': 0.2,
    'fb_paint_zoom': 1.0005,
    'color_cycle_speed': 0.12,
    'sat_curve_shadows': 1.3,
    'gamma': 1.15, 'contrast': 1.1,
}

# -----------------------------------------------------------------------
# PATCH 7: Oscillator Heat
# Coupled oscillator grid over a HOT-colored drift field.
# Perlin warp creates heat-haze shimmer. Feedback with horizontal
# drift gives molten metal flow. Warm color temperature throughout.
# -----------------------------------------------------------------------
patch7 = {**BASE_RESET,
    '_name': 'Oscillator Heat',
    'source_1': 'OSCILLATOR_GRID',
    'source_2': 'DRIFT_FIELD',
    'alpha_blend': 0.55,
    'osc_coupling': 0.7,
    'osc_noise': 0.08,
    'osc_freq_spread': 1.2,
    'osc_speed': 1.5,
    'osc_colormap': 11,   # HOT
    'osc_grid_size': 96,
    'drift_speed': 0.12,
    'drift_complexity': 4,
    'drift_scale': 6.0,
    'drift_viscosity': 0.997,
    'drift_injection': 0.025,
    'drift_colormap': 11,   # HOT
    'drift_color_speed': 0.2,
    'warp_type': 4,  # PERLIN
    'warp_speed': 20,
    'x_size': 15.0, 'y_size': 8.0,
    'x_speed': 2.0, 'y_speed': 0.3,
    'alpha': 0.65,
    'fb_paint_drift_x': 0.6,
    'fb_paint_drift_y': 0.05,
    'temporal_filter': 0.1,
    'color_temp': 0.35,
    'sat_curve_highlights': 1.6,
    'sat_curve_mids': 1.3,
    'gamma': 0.8, 'brightness': 5,
}

# -----------------------------------------------------------------------
# PATCH 8: DLA Crystal Garden
# DLA crystal growth luma-keyed over a tectonic Voronoi backdrop.
# Erosion adds rock-face texture. Echo glitch gives occasional
# time-smear bursts as the crystal suddenly shifts.
# -----------------------------------------------------------------------
patch8 = {**BASE_RESET,
    '_name': 'DLA Crystal Garden',
    'source_1': 'DLA',
    'source_2': 'VORONOI',
    'blend_mode': 1,   # LUMA_KEY
    'luma_threshold': 80,
    'luma_selection': 2,  # BLACK - show bright crystals on top
    'luma_blur': 3,
    'dla_num_particles': 150,
    'dla_stickiness': 0.85,
    'dla_spawn_radius': 1.4,
    'dla_particle_speed': 2,
    'dla_branch_bias': 0.1,
    'dla_fade': 0.998,
    'dla_crystal_r': 200, 'dla_crystal_g': 240, 'dla_crystal_b': 255,
    'dla_particle_r': 255, 'dla_particle_g': 255, 'dla_particle_b': 180,
    'voronoi_num_points': 40,
    'voronoi_relax_speed': 0.15,
    'voronoi_jitter': 1.2,
    'voronoi_show_edges': 1,
    'voronoi_show_points': 0,
    'voronoi_fill_cells': 1,
    'voronoi_edge_thickness': 1,
    'voronoi_colormap': 20,   # TURBO
    'voronoi_color_speed': 0.08,
    'voronoi_tectonic_speed': 0.3,
    'voronoi_tectonic_chaos': 0.5,
    'voronoi_edge_r': 30, 'voronoi_edge_g': 30, 'voronoi_edge_b': 60,
    'erosion_strength': 0.3,
    'erosion_scale': 4.0,
    'erosion_speed': 0.15,
    'erosion_octaves': 4,
    'erosion_sharpness': 0.5,
    'enable_echo': 1,
    'echo_probability': 0.04,
    'echo_buffer_size': 20,
    'echo_freeze_min': 3, 'echo_freeze_max': 8,
    'echo_blend_amount': 0.7,
    'false_color_strength': 0.2,
    'false_color_map': 17,   # CIVIDIS
    'color_temp': -0.2,
    'sat_curve_shadows': 0.7,
    'contrast': 1.15,
}

patches = [patch1, patch2, patch3, patch4, patch5, patch6, patch7, patch8]
names   = [p['_name'] for p in patches]

save_path = r'c:\Users\khend\Downloads\video_synth\save\saved_values.yaml'

with open(save_path, 'r') as f:
    data = yaml.safe_load(f) or {}

entries = data.get('entries', [])
before = len(entries)
for patch in patches:
    entries.append(patch)
data['entries'] = entries

with open(save_path, 'w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)

print(f'Appended {len(patches)} patches (entries {before+1}..{len(entries)})')
for i, n in enumerate(names, start=before+1):
    print(f'  [{i}] {n}')
