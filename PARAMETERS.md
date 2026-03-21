# Video Synth Parameter Reference

This document describes all parameters across every module. It is intended to guide the addition of `info` fields for in-app tooltips.

---

## Table of Contents

- [Plasma](#plasma)
- [Reaction Diffusion](#reaction-diffusion)
- [Drift Field](#drift-field)
- [Shaders (S1)](#shaders-s1)
- [Shaders 2 (S2)](#shaders-2-s2)
- [Voronoi](#voronoi)
- [Metaballs](#metaballs)
- [Moire](#moire)
- [Chladni](#chladni)
- [DLA (Diffusion-Limited Aggregation)](#dla-diffusion-limited-aggregation)
- [Physarum](#physarum)
- [Lenia](#lenia)
- [Fractal Zoom](#fractal-zoom)
- [Oscillator Grid](#oscillator-grid)
- [Harmonic Interference](#harmonic-interference)
- [Strange Attractor](#strange-attractor)
- [Color Effects](#color-effects)
- [Warp](#warp)
- [Feedback](#feedback)
- [Glitch](#glitch)
- [Erosion](#erosion)
- [Shapes](#shapes)
- [Pixels](#pixels)
- [PTZ (Pan/Tilt/Zoom)](#ptz-pantiltzoom)
- [Reflector](#reflector)
- [Lissajous](#lissajous)
- [Mixer](#mixer)
- [LFO (Low Frequency Oscillator)](#lfo-low-frequency-oscillator)
- [Audio Reactive](#audio-reactive)

---

## Plasma

Generates animated plasma-style patterns using layered sine wave functions.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Speed | `plasma_speed` | 0.01 | 10 | 1.0 | Overall animation speed of the plasma pattern |
| Distance | `plasma_distance` | 0.01 | 10 | 1.0 | Spatial scale/spread of the sine wave pattern |
| Color Speed | `plasma_color_speed` | 0.01 | 10 | 1.0 | Rate at which colors cycle through the palette |
| Flow Speed | `plasma_flow_speed` | 0.01 | 10 | 1.0 | Speed of directional flow/drift through the pattern |

---

## Reaction Diffusion

Simulates a two-chemical reaction-diffusion system (Gray-Scott model) producing organic-looking patterns.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Diffusion A | `da` | 0 | 2.0 | varies | Diffusion rate of chemical A; higher values spread it faster |
| Diffusion B | `db` | 0 | 2.0 | varies | Diffusion rate of chemical B; typically slower than A |
| Feed Rate | `feed` | 0 | 0.1 | varies | Rate at which chemical A is added to the system; controls pattern type |
| Kill Rate | `kill` | 0 | 0.1 | varies | Rate at which chemical B is removed; controls pattern density |
| Iterations/Frame | `iterations_per_frame` | 5 | 100 | 50 | Number of simulation steps computed per rendered frame |

---

## Drift Field

Animates a vector/flow field where particles or colors drift along evolving noise-based paths.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Speed | `drift_speed` | 0.01 | 2.0 | 0.15 | How fast the field evolves over time |
| Complexity | `drift_complexity` | 1 | 8 | 3 | Number of noise layers; higher values produce more tangled paths |
| Scale | `drift_scale` | 0.5 | 10.0 | 3.0 | Spatial scale of the noise field; larger values zoom out the pattern |
| Viscosity | `drift_viscosity` | 0.9 | 1.0 | 0.995 | How much the previous frame persists; near 1.0 creates long trails |
| Injection | `drift_injection` | 0.0 | 1.0 | 0.02 | Rate at which new color/energy is injected into the field |
| Colormap | `drift_colormap` | — | — | TWILIGHT | Color palette applied to the field intensity |
| Color Speed | `drift_color_speed` | 0.0 | 2.0 | 0.3 | Rate at which the colormap offset shifts over time |

---

## Shaders (S1)

First shader bank — applies GLSL-based generative visuals with a selectable shader type.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Shader Type | `s_type` | — | — | 0 | Selects which shader pattern to render |
| Zoom | `s_zoom` | 0.1 | 5.0 | 1.5 | Zoom level into the shader pattern |
| Distortion | `s_distortion` | 0.0 | 1.0 | 0.5 | Amount of spatial distortion applied to the pattern |
| Iterations | `s_iterations` | 1.0 | 10.0 | 4.0 | Number of fractal/fold iterations; higher = more detail |
| Color Shift | `s_color_shift` | 0.5 | 3.0 | 1.0 | Shifts the color mapping of the output |
| Brightness | `s_brightness` | 0.0 | 2.0 | 1.0 | Output brightness multiplier |
| Hue Shift | `s_hue_shift` | 0.0 | 7.0 | 0.0 | Rotates the hue of the shader output |
| Saturation | `s_saturation` | 0.0 | 2.0 | 1.0 | Saturation multiplier of the output |
| X Shift | `s_x_shift` | -5.0 | 5.0 | 0.0 | Horizontal pan/offset into the shader space |
| Y Shift | `s_y_shift` | -5.0 | 5.0 | 0.0 | Vertical pan/offset into the shader space |
| Rotation | `s_rotation` | -3.14 | 3.14 | 0.0 | Rotation of the shader coordinate space (radians) |
| Speed | `s_speed` | 0.0 | 2.0 | 1.0 | Animation speed of the shader |

---

## Shaders 2 (S2)

Second shader bank — a separate set of GLSL patterns with extended parameter range.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Shader Type | `s2_type` | — | — | 0 | Selects which shader pattern to render |
| Zoom | `s2_zoom` | 0.1 | 10.0 | 1.0 | Zoom level into the shader pattern |
| Distortion | `s2_distortion` | 0.0 | 2.0 | 0.5 | Amount of spatial distortion applied to the pattern |
| Iterations | `s2_iterations` | 1.0 | 20.0 | 6.0 | Number of fractal/fold iterations |
| Color Shift | `s2_color_shift` | 0.0 | 6.28 | 0.0 | Hue/phase offset for color mapping (0–2π) |
| Brightness | `s2_brightness` | 0.0 | 3.0 | 1.0 | Output brightness multiplier |
| Speed | `s2_speed` | 0.0 | 3.0 | 1.0 | Animation speed of the shader |
| Param A | `s2_param_a` | 0.0 | 10.0 | 3.0 | Shader-specific parameter A (meaning varies by type) |
| Param B | `s2_param_b` | 0.0 | 10.0 | 2.0 | Shader-specific parameter B (meaning varies by type) |

---

## Voronoi

Renders animated Voronoi diagrams with relaxation, tectonic drift, and color cycling.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Num Points | `voronoi_num_points` | 5 | 200 | 50 | Number of Voronoi seed points |
| Relax Speed | `voronoi_relax_speed` | 0.01 | 0.5 | 0.1 | Rate at which points drift toward cell centroids (Lloyd's relaxation) |
| Jitter | `voronoi_jitter` | 0.0 | 5.0 | 0.5 | Random perturbation added to point positions each frame |
| Show Edges | `voronoi_show_edges` | — | — | on | Toggle rendering of cell boundary lines |
| Show Points | `voronoi_show_points` | — | — | on | Toggle rendering of seed point dots |
| Fill Cells | `voronoi_fill_cells` | — | — | on | Toggle filling cells with color |
| Edge Thickness | `voronoi_edge_thickness` | 1 | 5 | 2 | Pixel width of cell boundary lines |
| Point Size | `voronoi_point_size` | 2 | 10 | 5 | Pixel radius of seed point dots |
| Edge R | `voronoi_edge_r` | 0 | 255 | 255 | Red channel of edge color |
| Edge G | `voronoi_edge_g` | 0 | 255 | 255 | Green channel of edge color |
| Edge B | `voronoi_edge_b` | 0 | 255 | 255 | Blue channel of edge color |
| Colormap | `voronoi_colormap` | — | — | 0 | Color palette used to fill cells |
| Color Speed | `voronoi_color_speed` | 0.0 | 2.0 | 0.2 | Rate at which colormap offset shifts over time |
| Tectonic Speed | `voronoi_tectonic_speed` | 0.0 | 3.0 | 0.0 | Speed of large-scale directional drift across all points |
| Tectonic Chaos | `voronoi_tectonic_chaos` | 0.0 | 1.0 | 0.3 | Randomness in tectonic drift direction per point |

---

## Metaballs

Renders smooth organic blobs (metaballs) that merge and separate.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Num Metaballs | `num_metaballs` | 2 | 10 | 5 | Number of metaball blobs |
| Min Radius | `min_radius` | 20 | 100 | 40 | Minimum radius (px) a metaball can have |
| Max Radius | `max_radius` | 40 | 200 | 80 | Maximum radius (px) a metaball can have |
| Radius Multiplier | `radius_multiplier` | 1.0 | 3.0 | 1.0 | Global scale applied to all metaball radii |
| Max Speed | `max_speed` | 1 | 10 | 3 | Maximum velocity of each metaball |
| Speed Multiplier | `speed_multiplier` | 1.0 | 3.0 | 1.0 | Global scale applied to all metaball velocities |
| Threshold | `threshold` | 0.5 | 3.0 | 1.6 | Isosurface threshold; controls where blob surfaces appear |
| Smooth Coloring Max | `smooth_coloring_max_field` | 1.0 | 3.0 | 1.5 | Upper field value used to normalize smooth color gradient |
| Skew Angle | `metaball_skew_angle` | 0.0 | 360.0 | 0.0 | Direction of coordinate skew applied to the canvas |
| Skew Intensity | `metaball_skew_intensity` | 0.0 | 1.0 | 0.0 | Strength of the skew distortion |
| Zoom | `metaball_zoom` | 1.0 | 3.0 | 1.0 | Zoom level into the metaball field |
| Colormap | `metaball_colormap` | — | — | JET | Color palette used to shade blobs |
| Feedback | `metaballs_feedback` | 0.0 | 1.0 | 0.95 | Alpha blend of previous frame over current; creates trails |
| Render Scale | `metaballs_render_scale` | 0.25 | 1.0 | 0.25 | Internal resolution as a fraction of output size; lower = faster |

---

## Moire

Creates interference patterns by overlaying two independently controllable grids or radial patterns.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Blend Mode | `moire_blend` | — | — | 0 | How the two layers are composited (e.g. multiply, XOR) |
| Pattern 1 Type | `moire_type_1` | — | — | 0 | Shape of layer 1 (lines, circles, dots, etc.) |
| Spatial Freq 1 | `spatial_freq_1` | 0.01 | 25 | 10.0 | Spatial frequency (line density) of layer 1 |
| Angle 1 | `angle_1` | 0 | 360 | 90.0 | Rotation angle of layer 1 (degrees) |
| Zoom 1 | `zoom_1` | 0.05 | 1.5 | 1.0 | Zoom level of layer 1 |
| Center X 1 | `moire_center_x_1` | 0 | width | center | Horizontal center point of layer 1 |
| Center Y 1 | `moire_center_y_1` | 0 | height | center | Vertical center point of layer 1 |
| Pattern 2 Type | `moire_type_2` | — | — | 0 | Shape of layer 2 |
| Spatial Freq 2 | `spatial_freq_2` | 0.01 | 25 | 1.0 | Spatial frequency of layer 2 |
| Angle 2 | `angle_2` | 0 | 360 | 0.0 | Rotation angle of layer 2 (degrees) |
| Zoom 2 | `zoom_2` | 0.05 | 1.5 | 1.0 | Zoom level of layer 2 |
| Center X 2 | `moire_center_x_2` | 0 | width | center | Horizontal center point of layer 2 |
| Center Y 2 | `moire_center_y_2` | 0 | height | center | Vertical center point of layer 2 |

---

## Chladni

Simulates Chladni figures — sand particles settling at the nodes of a vibrating plate.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Freq M | `chladni_freq_m` | 1 | 20 | 5.0 | X-axis modal frequency of the vibrating plate |
| Freq N | `chladni_freq_n` | 1 | 20 | 3.0 | Y-axis modal frequency of the vibrating plate |
| Amplitude | `chladni_amplitude` | 0.1 | 2.0 | 1.0 | Vibration amplitude; affects how strongly particles are repelled from antinodes |
| Speed | `chladni_speed` | 0.0 | 2.0 | 0.5 | Rate at which the wave pattern evolves |
| Blend | `chladni_blend` | 0.0 | 1.0 | 0.5 | Mix between two wave modes |
| Num Particles | `chladni_particles` | 1000 | 50000 | 10000 | Total number of simulated particles |
| Particle Speed | `chladni_particle_speed` | 0.1 | 5.0 | 1.0 | Velocity at which particles move toward nodal lines |
| Friction | `chladni_friction` | 0.8 | 0.99 | 0.95 | Velocity damping per frame; higher = more retained momentum |
| Show Wave | `chladni_show_wave` | — | — | on | Toggle rendering the underlying wave field |
| Colormap | `chladni_colormap` | — | — | 2 | Color palette applied to the wave field visualization |
| Particle R | `chladni_particle_r` | 0 | 255 | 255 | Red channel of particle color |
| Particle G | `chladni_particle_g` | 0 | 255 | 255 | Green channel of particle color |
| Particle B | `chladni_particle_b` | 0 | 255 | 200 | Blue channel of particle color |

---

## DLA (Diffusion-Limited Aggregation)

Grows fractal crystal structures by randomly walking particles that stick on contact.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Num Particles | `dla_num_particles` | 10 | 500 | 100 | Number of active random-walking particles |
| Stickiness | `dla_stickiness` | 0.1 | 1.0 | 1.0 | Probability a particle sticks when it touches the crystal |
| Spawn Radius | `dla_spawn_radius` | 1.1 | 2.0 | 1.3 | How far from the crystal edge new particles spawn (as a ratio) |
| Particle Speed | `dla_particle_speed` | 1 | 10 | 3 | Steps per frame each particle moves |
| Branch Bias | `dla_branch_bias` | -1.0 | 1.0 | 0.0 | Directional bias in particle walk; negative = inward, positive = outward |
| Fade | `dla_fade` | 0.0 | 1.0 | 0.99 | How quickly old crystal deposits fade; near 1.0 = persistent |
| Crystal R/G/B | `dla_crystal_r/g/b` | 0 | 255 | 100/200/255 | RGB color of the crystal structure |
| Particle R/G/B | `dla_particle_r/g/b` | 0 | 255 | 255/255/255 | RGB color of active walking particles |
| Reset | `dla_reset` | — | — | off | Toggle to clear the crystal and restart the simulation |

---

## Physarum

Simulates slime mold (Physarum polycephalum) agent behavior — agents deposit trail and steer toward chemical gradients.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Num Agents | `phys_num_agents` | 1000 | 10000 | 1000 | Number of simulated slime mold agents |
| Sensor Angle | `phys_sensor_angle_spacing` | 0.0 | π/2 | π/8 | Angular offset of the forward sensors from the heading |
| Sensor Distance | `phys_sensor_distance` | 1 | 20 | 9 | How far ahead each sensor samples the trail map |
| Turn Angle | `phys_turn_angle` | 0.0 | π/2 | π/4 | Maximum turn angle per step when steering |
| Step Distance | `phys_step_distance` | 1 | 10 | 1 | Pixels each agent moves per simulation step |
| Decay Factor | `phys_decay_factor` | 0.0 | 1.0 | 0.1 | Rate at which trail pheromone evaporates each frame |
| Diffuse Factor | `phys_diffuse_factor` | 0.0 | 1.0 | 0.5 | Amount of trail blur/diffusion applied each frame |
| Deposit Amount | `phys_deposit_amount` | 0.1 | 5.0 | 1.0 | How much pheromone each agent deposits per step |
| Grid Resolution | `phys_grid_res_scale` | 0.1 | 1.0 | 0.5 | Internal grid resolution as fraction of output (lower = faster) |
| Wrap Around | `phys_wrap_around` | — | — | on | Toggle whether agents wrap at canvas edges |
| Trail R/G/B | `phys_trail_r/g/b` | 0 | 255 | 0/255/0 | RGB color of the pheromone trail |
| Agent R/G/B | `phys_agent_r/g/b` | 0 | 255 | 255/0/0 | RGB color of rendered agents |
| Agent Size | `phys_agent_size` | 1 | 5 | 1 | Pixel size of each rendered agent dot |

---

## Lenia

A continuous cellular automaton producing smooth, life-like moving organisms.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| dt | `lenia_dt` | 0.01 | 0.5 | 0.1 | Time step per frame; smaller = more stable but slower evolution |
| Mu | `lenia_mu` | 0.05 | 0.5 | 0.15 | Center of the growth function; where cells grow most strongly |
| Sigma | `lenia_sigma` | 0.005 | 0.1 | 0.017 | Width of the growth function bell curve |
| Radius | `lenia_radius` | 5 | 30 | 13 | Neighborhood radius each cell samples |
| Colormap | `lenia_colormap` | — | — | INFERNO | Color palette for cell values |
| Seed Density | `lenia_seed_density` | 0.01 | 0.5 | 0.15 | Initial fraction of cells populated when reseeding |

---

## Fractal Zoom

Continuously zooms into a Mandelbrot/Julia-style fractal while drifting through parameter space.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Zoom Speed | `fractal_zoom_speed` | 0.0 | 1.0 | 0.1 | Rate at which the view zooms in |
| Drift Speed | `fractal_drift_speed` | 0.0 | 1.0 | 0.2 | Speed of lateral drift through the fractal plane |
| Max Iterations | `fractal_max_iter` | 20 | 200 | 64 | Maximum iteration count; higher = more detail but slower |
| Color Speed | `fractal_color_speed` | 0.0 | 2.0 | 0.5 | Rate at which the colormap offset cycles |
| Colormap | `fractal_colormap` | — | — | TWILIGHT_SHIFTED | Color palette applied to iteration depth |

---

## Oscillator Grid

A grid of coupled oscillators rendered as a phase/amplitude field.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Coupling | `osc_coupling` | 0.0 | 2.0 | 0.5 | Strength of phase coupling between neighboring oscillators |
| Noise | `osc_noise` | 0.0 | 0.5 | 0.05 | Random perturbation added to each oscillator per frame |
| Freq Spread | `osc_freq_spread` | 0.0 | 2.0 | 0.5 | Variance in natural frequencies across the grid |
| Speed | `osc_speed` | 0.1 | 5.0 | 1.0 | Overall simulation speed multiplier |
| Colormap | `osc_colormap` | — | — | HSV | Color palette for phase visualization |
| Grid Size | `osc_grid_size` | 32 | 256 | 80 | Number of oscillators per side (N×N grid) |

---

## Harmonic Interference

Layers multiple sinusoidal waves at different frequencies and orientations to create interference patterns.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Num Layers | `hi_num_layers` | 2 | 8 | 5 | Number of overlaid wave layers |
| Base Freq | `hi_base_freq` | 0.5 | 20.0 | 4.0 | Spatial frequency of the first/base wave layer |
| Freq Spread | `hi_freq_spread` | 0.0 | 2.0 | 0.5 | How much each successive layer's frequency differs |
| Drift Speed | `hi_drift_speed` | 0.0 | 2.0 | 0.3 | Speed at which wave phases drift over time |
| Rotation Speed | `hi_rotation_speed` | 0.0 | 1.0 | 0.1 | Rate at which wave orientations rotate |
| Color Speed | `hi_color_speed` | 0.0 | 2.0 | 0.2 | Rate at which the colormap offset shifts |
| Colormap | `hi_colormap` | — | — | TWILIGHT | Color palette for rendering the interference field |

---

## Strange Attractor

Plots the trajectory of a strange attractor system (Lorenz, Clifford, De Jong, Aizawa, Thomas).

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Attractor Type | `attractor_type` | — | — | 0 | Selects which attractor system to simulate |
| dt | `attractor_dt` | 0.001 | 0.05 | 0.01 | Integration time step; smaller = more accurate but slower growth |
| Num Steps | `attractor_num_steps` | 1 | 50 | 10 | Steps computed per rendered frame |
| Scale | `attractor_scale` | 1.0 | 20.0 | 5.0 | Zoom level / coordinate scaling |
| Line Width | `attractor_line_width` | 1 | 5 | 1 | Width of the drawn trajectory line |
| Fade | `attractor_fade` | 0.0 | 1.0 | 0.95 | Alpha of previous frame overlay; near 1.0 = long persistent trails |
| Trail R/G/B | `attractor_r/g/b` | 0 | 255 | 255/255/255 | RGB color of the attractor trail |
| Morph Speed | `attractor_morph_speed` | 0.0 | 1.0 | 0.0 | Rate at which attractor parameters slowly drift/morph |
| **Lorenz** | | | | | |
| Sigma | `lorenz_sigma` | 1.0 | 20.0 | 10.0 | Lorenz σ (Prandtl number) |
| Rho | `lorenz_rho` | 1.0 | 50.0 | 28.0 | Lorenz ρ (Rayleigh number); near 28 produces the classic butterfly |
| Beta | `lorenz_beta` | 0.1 | 5.0 | 2.667 | Lorenz β (geometric factor) |
| **Clifford** | | | | | |
| A / B / C / D | `clifford_a/b/c/d` | -3.0 | 3.0 | varies | Clifford attractor coefficients |
| **De Jong** | | | | | |
| A / B / C / D | `dejong_a/b/c/d` | -3.0 | 3.0 | varies | De Jong attractor coefficients |
| **Aizawa** | | | | | |
| A–F | `aizawa_a/b/c/d/e/f` | varies | varies | varies | Aizawa attractor coefficients |
| **Thomas** | | | | | |
| B | `thomas_b` | 0.1 | 0.3 | 0.208 | Thomas cyclically symmetric attractor dissipation coefficient |

---

## Color Effects

Post-processing effects applied to the output image's color properties.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Hue Shift | `hue_shift` | 0 | 180 | 0 | Rotates all hues by this amount (OpenCV hue units, 0–180) |
| Saturation Shift | `sat_shift` | 0 | 255 | 0 | Adds to the saturation channel |
| Value Shift | `val_shift` | 0 | 255 | 0 | Adds to the value/brightness channel |
| Posterize Levels | `posterize_levels` | 0 | 100 | 0 | Number of discrete color levels per channel; 0 = disabled |
| Num Hues | `num_hues` | 2 | 10 | 8 | Number of hue steps when posterize is active |
| Val Threshold | `val_threshold` | 0 | 255 | 0 | Brightness threshold below which hue shift is applied |
| Val Hue Shift | `val_hue_shift` | 0 | 255 | 0 | Hue shift applied to pixels below the value threshold |
| Solarize Threshold | `solarize_threshold` | 0 | 100 | 0 | Pixels above this brightness get their values inverted |
| Hue Invert Angle | `hue_invert_angle` | 0 | 360 | 0 | Target hue to selectively invert |
| Hue Invert Strength | `hue_invert_strength` | 0.0 | 1.0 | 0.0 | How strongly the hue inversion is applied |
| Contrast | `contrast` | 0.5 | 3.0 | 1.0 | Contrast multiplier around the midpoint |
| Brightness | `brightness` | 0 | 100 | 0 | Additive brightness offset |
| Gamma | `gamma` | 0.1 | 3.0 | 1.0 | Gamma correction curve; < 1 brightens shadows |
| Highlight Compression | `highlight_compression` | 0.0 | 1.0 | 0.0 | Rolls off highlights to prevent clipping |
| Color Cycle Speed | `color_cycle_speed` | 0.0 | 5.0 | 0.0 | Speed at which all hues continuously rotate |
| Color Cycle Bands | `color_cycle_bands` | 1 | 8 | 3 | Number of hue bands used in color cycling |
| Channel Mix RR/RG/RB | `ch_mix_rr/rg/rb` | 0.0 | 2.0 | 1/0/0 | How much R, G, B contribute to the output Red channel |
| Channel Mix GR/GG/GB | `ch_mix_gr/gg/gb` | 0.0 | 2.0 | 0/1/0 | How much R, G, B contribute to the output Green channel |
| Channel Mix BR/BG/BB | `ch_mix_br/bg/bb` | 0.0 | 2.0 | 0/0/1 | How much R, G, B contribute to the output Blue channel |
| Color Bitcrush | `color_bitcrush` | 1 | 8 | 8 | Reduces color bit depth; lower = more posterized/banded look |
| Hue Scatter | `hue_scatter` | 0.0 | 1.0 | 0.0 | Adds random per-pixel hue variation |
| Duotone Strength | `duotone_strength` | 0.0 | 1.0 | 0.0 | Blends image toward a two-hue palette |
| Duotone Hue Lo | `duotone_hue_lo` | 0 | 180 | 120 | Hue assigned to the shadow/low end of the duotone |
| Duotone Hue Hi | `duotone_hue_hi` | 0 | 180 | 10 | Hue assigned to the highlight/high end of the duotone |
| Ch R / G / B | `ch_r/g/b` | 0.0 | 1.0 | 1.0 | Per-channel gain multiplier (simple RGB scaling) |
| Chromatic Ab X | `chroma_ab_x` | 0 | 30 | 0 | Horizontal chromatic aberration offset between channels |
| Chromatic Ab Y | `chroma_ab_y` | 0 | 30 | 0 | Vertical chromatic aberration offset between channels |
| Color Temp | `color_temp` | -1.0 | 1.0 | 0.0 | Shifts color temperature; negative = cool/blue, positive = warm/orange |
| Sat Shadows | `sat_curve_shadows` | 0.0 | 3.0 | 1.0 | Saturation multiplier for shadow tones |
| Sat Midtones | `sat_curve_mids` | 0.0 | 3.0 | 1.0 | Saturation multiplier for midtones |
| Sat Highlights | `sat_curve_highlights` | 0.0 | 3.0 | 1.0 | Saturation multiplier for highlight tones |
| False Color Strength | `false_color_strength` | 0.0 | 1.0 | 0.0 | Blends a false-color (luma-mapped) version over the image |
| False Color Map | `false_color_map` | — | — | INFERNO | Colormap used for the false color overlay |
| Invert Strength | `invert_strength` | 0.0 | 1.0 | 0.0 | Blends an inverted version of the image; 1.0 = fully inverted |

---

## Warp

Applies spatial displacement/warping to the image using various noise and feedback methods.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Warp Type | `warp_type` | — | — | 0 | Selects the warp method (noise, feedback, displacement, etc.) |
| Angle Amount | `warp_angle_amt` | 0 | 360 | 30 | Angular displacement amount for rotation-based warps |
| Radius Amount | `warp_radius_amt` | 0 | 360 | 30 | Radial displacement amount for radial warps |
| Warp Speed | `warp_speed` | 0 | 100 | 10 | Rate at which the warp field animates |
| Use Fractal | `warp_use_fractal` | — | — | off | Toggle fractal (multi-octave) noise for warp field |
| Octaves | `warp_octaves` | 1 | 8 | 4 | Number of noise octaves when fractal mode is on |
| Gain | `warp_gain` | 0.0 | 1.0 | 0.5 | Amplitude falloff per octave in fractal noise |
| Lacunarity | `warp_lacunarity` | 1.0 | 4.0 | 2.0 | Frequency multiplier per octave in fractal noise |
| X Speed | `x_speed` | 0.0 | 100.0 | 1.0 | Horizontal drift speed of the warp field |
| X Size | `x_size` | 0.25 | 100.0 | 20.0 | Horizontal scale of the noise warp |
| Y Speed | `y_speed` | 0.0 | 10.0 | 1.0 | Vertical drift speed of the warp field |
| Y Size | `y_size` | 0.25 | 100.0 | 10.0 | Vertical scale of the noise warp |
| FB Warp Decay | `fb_warp_decay` | 0.0 | 1.0 | 0.95 | How quickly the feedback warp field decays |
| FB Warp Strength | `fb_warp_strength` | 0.0 | 50.0 | 5.0 | Intensity of feedback-driven displacement |
| FB Warp Freq | `fb_warp_freq` | 0.1 | 20.0 | 3.0 | Frequency of the noise used in feedback warp |
| Disp Strength | `disp_strength` | 0.0 | 30.0 | 5.0 | Magnitude of displacement map warping |
| Disp Decay | `disp_decay` | 0.0 | 1.0 | 0.92 | Decay rate of the displacement map |
| Disp Blur | `disp_blur` | 1 | 15 | 5 | Blur kernel size applied to the displacement map |
| Conv Rise Speed | `conv_rise_speed` | 0.0 | 10.0 | 2.0 | Speed at which convection field intensity rises |
| Conv Diffusion | `conv_diffusion` | 0.0 | 1.0 | 0.5 | Spatial diffusion rate of the convection field |
| Conv Turbulence | `conv_turbulence` | 0.0 | 1.0 | 0.3 | Amount of turbulent noise in the convection field |
| Conv Decay | `conv_decay` | 0.0 | 1.0 | 0.95 | Decay rate of the convection field |
| RD Warp Strength | `rd_warp_strength` | 0.0 | 30.0 | 10.0 | Strength of reaction-diffusion driven warping |
| RD Warp Feed | `rd_warp_feed` | 0.01 | 0.1 | 0.055 | Feed rate for the internal RD warp simulation |
| RD Warp Kill | `rd_warp_kill` | 0.03 | 0.08 | 0.062 | Kill rate for the internal RD warp simulation |
| RD Warp Speed | `rd_warp_speed` | 0.1 | 5.0 | 1.0 | Simulation speed of the RD warp field |

---

## Feedback

Controls how previous frames are blended back into the current frame.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Alpha | `alpha` | 0.0 | 1.0 | 0.0 | Blend strength of the feedback frame over the current frame |
| Temporal Filter | `temporal_filter` | 0.0 | 1.0 | 0.0 | Low-pass filter applied across frames to smooth flickering |
| Luma Threshold | `feedback_luma_threshold` | 0 | 255 | 0 | Feedback is only applied to pixels above this brightness |
| Luma Mode | `luma_mode` | — | — | WHITE | Whether the threshold selects bright or dark pixels for feedback |
| Frame Skip | `frame_skip` | 0 | 10 | 0 | Number of frames to skip when reading back the feedback buffer |
| Buffer Frame Select | `buffer_frame_select` | -1 | 20 | -1 | Selects a specific past frame from the buffer; -1 = latest |
| Buffer Frame Blend | `buffer_frame_blend` | 0.0 | 1.0 | 0.0 | Mix between the live feed and the selected buffer frame |
| Prev Frame Scale | `prev_frame_scale` | 90 | 110 | 100 | Scales the previous frame before blending (100 = no change) |
| Buffer Size | `buffer_size` | 0 | max | 0 | Number of past frames held in memory |
| Paint Drift X | `fb_paint_drift_x` | -5.0 | 5.0 | 0.0 | Horizontal pixel offset applied to the feedback frame each cycle |
| Paint Drift Y | `fb_paint_drift_y` | -5.0 | 5.0 | 0.0 | Vertical pixel offset applied to the feedback frame each cycle |
| Paint Rotation | `fb_paint_rotation` | -2.0 | 2.0 | 0.0 | Rotation applied to the feedback frame each cycle (degrees) |
| Paint Zoom | `fb_paint_zoom` | 0.99 | 1.01 | 1.0 | Subtle zoom applied to the feedback frame each cycle |

---

## Glitch

Simulates digital video glitch artifacts including pixel shifts, color splits, block corruption, and slit-scan effects.

### General

| Parameter | Key | Default | Description |
|-----------|-----|---------|-------------|
| Pixel Shift | `enable_pixel_shift` | off | Randomly shifts horizontal scanline slices |
| Color Split | `enable_color_split` | off | Offsets R/G/B channels from each other |
| Block Corruption | `enable_block_corruption` | off | Replaces random blocks with noise or solid color |
| Random Rectangles | `enable_random_rectangles` | off | Draws random colored rectangles over the image |
| H-Scroll Freeze | `enable_horizontal_scroll_freeze` | off | Freezes a horizontal band and scrolls it |
| Duration Frames | `glitch_duration_frames` | 1–300 | 60 | How many frames a triggered glitch event lasts |
| Intensity Max | `glitch_intensity_max` | 0–100 | 50 | Maximum displacement/intensity of glitch artifacts |
| Block Size Max | `glitch_block_size_max` | 0–200 | 60 | Maximum size of corrupted block regions |
| Band Div | `glitch_band_div` | 1–10 | 5 | Number of horizontal bands scanned for glitching |
| Num Glitches | `num_glitches` | 0–100 | 0 | Number of simultaneous glitch events |
| Glitch Size | `glitch_size` | 1–100 | 0 | Size of each individual glitch artifact |

### Slit-Scan

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Enable | `enable_slitscan` | — | — | off | Toggle slit-scan effect |
| Direction | `slitscan_direction` | — | — | 0 | Horizontal or vertical scan direction |
| Slice Width | `slitscan_slice_width` | 1 | 50 | 5 | Width (px) of each captured time slice |
| Time Offset | `slitscan_time_offset` | 1 | 60 | 10 | Frame delay between consecutive slices |
| Speed | `slitscan_speed` | 0.1 | 10.0 | 1.0 | Rate at which the scan position advances |
| Reverse | `slitscan_reverse` | — | — | off | Toggle to scan in the opposite direction |
| Buffer Size | `slitscan_buffer_size` | 10 | 120 | 60 | Number of past frames held for slit-scan compositing |
| Blend Mode | `slitscan_blend_mode` | — | — | 0 | How slices are composited onto the canvas |
| Blend Alpha | `slitscan_blend_alpha` | 0.0 | 1.0 | 1.0 | Opacity of each composited slice |
| Position Offset | `slitscan_position_offset` | -100 | 100 | 0 | Shifts the scan start position |
| Wobble Amount | `slitscan_wobble_amount` | 0 | 50 | 0 | Oscillating positional wobble applied to the scan |
| Wobble Freq | `slitscan_wobble_freq` | 0.1 | 10.0 | 1.0 | Frequency of the position wobble |

### Echo

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Enable | `enable_echo` | — | — | off | Toggle echo/stutter glitch |
| Probability | `echo_probability` | 0.0 | 1.0 | 0.1 | Chance per frame that an echo event triggers |
| Buffer Size | `echo_buffer_size` | 5 | 60 | 30 | Number of past frames held for echo |
| Freeze Min | `echo_freeze_min` | 1 | 30 | 2 | Minimum duration (frames) of a freeze event |
| Freeze Max | `echo_freeze_max` | 2 | 60 | 10 | Maximum duration (frames) of a freeze event |
| Blend Amount | `echo_blend_amount` | 0.0 | 1.0 | 1.0 | Opacity of the echoed frame when composited |

---

## Erosion

Applies a terrain-erosion-style noise displacement to the image.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Strength | `erosion_strength` | 0.0 | 1.0 | 0.0 | Overall intensity of the erosion effect; 0 = disabled |
| Scale | `erosion_scale` | 1.0 | 10.0 | 3.0 | Spatial scale of the erosion noise pattern |
| Speed | `erosion_speed` | 0.0 | 2.0 | 0.2 | Rate at which the erosion pattern evolves |
| Octaves | `erosion_octaves` | 1 | 6 | 4 | Number of noise octaves; higher = more fine detail |
| Sharpness | `erosion_sharpness` | 0.0 | 1.0 | 0.3 | Sharpens/ridges the erosion pattern |

---

## Shapes

Overlays procedural geometric shapes onto the output frame.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Shape Type | `shape_type` | — | — | NONE | Selects which shape to render |
| Line Hue | `line_hue` | 0 | 179 | 0 | Hue of the shape outline |
| Line Sat | `line_sat` | 0 | 255 | 255 | Saturation of the shape outline |
| Line Val | `line_val` | 0 | 255 | 255 | Brightness of the shape outline |
| Line Weight | `line_weight` | 1 | 20 | 2 | Stroke width (px) of the shape outline |
| Line Opacity | `line_opacity` | 0.0 | 1.0 | 0.66 | Transparency of the shape outline |
| Size Multiplier | `size_multiplier` | 0.1 | 10.0 | 0.9 | Scale of the shape relative to canvas |
| Aspect Ratio | `aspect_ratio` | 0.1 | 10.0 | 1.0 | Width-to-height ratio of the shape |
| Rotation Angle | `rotation_angle` | 0 | 360 | 0 | Rotation of the shape (degrees) |
| X Shift | `shape_x_shift` | -width | width | center | Horizontal position offset |
| Y Shift | `shape_y_shift` | -height | height | center | Vertical position offset |
| Grid X | `multiply_grid_x` | 1 | 10 | 2 | Number of columns when tiling the shape in a grid |
| Grid Y | `multiply_grid_y` | 1 | 10 | 2 | Number of rows when tiling the shape in a grid |
| Grid Pitch X | `grid_pitch_x` | 0 | width | 100 | Horizontal spacing between grid tiles (px) |
| Grid Pitch Y | `grid_pitch_y` | 0 | height | 100 | Vertical spacing between grid tiles (px) |
| Fill Hue | `fill_hue` | 0 | 179 | 120 | Hue of the shape fill |
| Fill Sat | `fill_sat` | 0 | 255 | 100 | Saturation of the shape fill |
| Fill Val | `fill_val` | 0 | 255 | 255 | Brightness of the shape fill |
| Fill Opacity | `fill_opacity` | 0.0 | 1.0 | 0.25 | Transparency of the fill |
| Canvas Rotation | `canvas_rotation` | 0 | 360 | 0 | Rotates the entire shapes layer |

---

## Pixels

Controls sharpening, blurring, and noise applied at the pixel level.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Sharpen Type | `sharpen_type` | — | — | 0 | Sharpening algorithm (unsharp mask, laplacian, etc.) |
| Sharpen Intensity | `sharpen_intensity` | 1.0 | 8.0 | 4.0 | Strength of the sharpening effect |
| Mask Blur | `mask_blur` | 1 | 10 | 5 | Blur radius used to generate the unsharp mask |
| K Size | `k_size` | 0 | 11 | 3 | Kernel size for various pixel operations |
| Blur Type | `blur_type` | — | — | 0 | Blur algorithm (gaussian, box, median, etc.) |
| Blur Kernel Size | `blur_kernel_size` | 1 | 100 | 1 | Kernel size of the blur (forced odd); 1 = no blur |
| Noise Type | `noise_type` | — | — | NONE | Type of noise to overlay (Gaussian, salt-and-pepper, etc.) |
| Noise Intensity | `noise_intensity` | 0.0 | 1.0 | 0.1 | Strength of the noise overlay |

---

## PTZ (Pan/Tilt/Zoom)

Applies pan, tilt, zoom, and rotation transforms to the output — simulating a camera move.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| X Shift | `x_shift` | -width | width | 0 | Horizontal pan offset (px) |
| Y Shift | `y_shift` | -height | height | 0 | Vertical tilt offset (px) |
| Zoom | `zoom` | 0.75 | 3.0 | 1.0 | Zoom level; > 1 zooms in |
| Rotation | `r_shift` | -360 | 360 | 0.0 | Rotation in degrees |
| Prev X Shift | `prev_x_shift` | -width | width | 0 | X shift applied to the previous-frame layer |
| Prev Y Shift | `prev_y_shift` | -height | height | 0 | Y shift applied to the previous-frame layer |
| Prev Zoom | `prev_zoom` | 0.75 | 3.0 | 1.0 | Zoom applied to the previous-frame layer |
| Prev Rotation | `prev_r_shift` | -360 | 360 | 0.0 | Rotation applied to the previous-frame layer |
| Prev CX | `prev_cx` | -width/2 | width/2 | 0 | Center X for previous-frame transform |
| Prev CY | `prev_cy` | -height/2 | height/2 | 0 | Center Y for previous-frame transform |
| Polar X | `polar_x` | -width/2 | width/2 | 0 | X origin for polar coordinate transform |
| Polar Y | `polar_y` | -height/2 | height/2 | 0 | Y origin for polar coordinate transform |
| Polar Radius | `polar_radius` | 0.1 | 100 | 1.0 | Radius scale for polar transform |

---

## Reflector

Applies mirror/kaleidoscope reflections to the image.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Reflection Mode | `reflection_mode` | — | — | NONE | Selects the reflection type (horizontal, vertical, radial, etc.) |
| Segments | `reflector_segments` | 0 | 10 | 0 | Number of kaleidoscope segments for radial mode |
| Zoom | `reflector_z` | 0.5 | 2.0 | 1.0 | Zoom level applied before reflection |
| Rotation | `reflector_r` | -360 | 360 | 0.0 | Rotation of the canvas before reflection (degrees) |

---

## Lissajous

Draws an animated Lissajous figure as a colored curve overlay.

| Parameter | Key | Min | Max | Default | Description |
|-----------|-----|-----|-----|---------|-------------|
| Amplitude X | `lissajous_amp_x` | 0.0 | 1.0 | 0.4 | Horizontal amplitude as a fraction of canvas size |
| Amplitude Y | `lissajous_amp_y` | 0.0 | 1.0 | 0.4 | Vertical amplitude as a fraction of canvas size |
| Freq X | `lissajous_freq_x` | 1 | 12 | 3 | Horizontal frequency ratio |
| Freq Y | `lissajous_freq_y` | 1 | 12 | 2 | Vertical frequency ratio |
| Phase | `lissajous_phase` | 0.0 | 1.0 | 0.25 | Phase offset between X and Y oscillations (as fraction of 2π) |
| Speed | `lissajous_speed` | 0.0 | 2.0 | 0.5 | Rate at which the phase evolves |
| Num Points | `lissajous_points` | 100 | 5000 | 1000 | Number of points sampled along the curve |
| Line Mode | `lissajous_line_mode` | — | — | 1 | Whether to draw continuous lines or individual dots |
| Thickness | `lissajous_thickness` | 1 | 10 | 2 | Stroke width of the curve |
| Hue Start | `lissajous_hue_start` | 0 | 180 | 0 | Starting hue when not in rainbow mode |
| Hue Range | `lissajous_hue_range` | 0 | 180 | 60 | Range of hues spanned along the curve |
| Saturation | `lissajous_saturation` | 0 | 255 | 255 | Saturation of the curve color |
| Brightness | `lissajous_brightness` | 0 | 255 | 255 | Brightness of the curve color |
| Rainbow Mode | `lissajous_rainbow` | — | — | on | Colors cycle through the full hue wheel along the curve |
| Harmonic Strength | `lissajous_harmonic` | 0.0 | 1.0 | 0.0 | Adds a secondary harmonic frequency to the curve |
| Harmonic Freq | `lissajous_harm_freq` | 2 | 8 | 3 | Frequency multiplier for the harmonic component |

---

## Mixer

Controls which two animation sources are combined and how they are blended.

| Parameter | Key | Description |
|-----------|-----|-------------|
| Source 1 | `source_1` | Selects the animation/source for layer 1 |
| Source 2 | `source_2` | Selects the animation/source for layer 2 |
| Video File Src 1 | `video_file_src1` | Selects a video file as source 1 |
| Video File Src 2 | `video_file_src2` | Selects a video file as source 2 |
| Image File Src 1 | `image_file_src1` | Selects an image file as source 1 |
| Image File Src 2 | `image_file_src2` | Selects an image file as source 2 |
| Blend Mode | `blend_mode` | How the two sources are combined (alpha, luma key, chroma key) |
| Luma Threshold | `luma_threshold` | Brightness level that defines the luma key boundary (0–255) |
| Luma Selection | `luma_selection` | Whether to key out bright or dark pixels |
| Luma Blur | `luma_blur` | Feathers the edges of the luma key mask |
| Upper Hue | `upper_hue` | Upper bound of the chroma key hue range |
| Upper Sat | `upper_sat` | Upper bound of the chroma key saturation range |
| Upper Val | `upper_val` | Upper bound of the chroma key value range |
| Lower Hue | `lower_hue` | Lower bound of the chroma key hue range |
| Lower Sat | `lower_sat` | Lower bound of the chroma key saturation range |
| Lower Val | `lower_val` | Lower bound of the chroma key value range |
| Alpha Blend | `alpha_blend` | Mix ratio between source 1 and source 2 (0 = all S1, 1 = all S2) |
| Swap Sources | `swap_sources` | Toggle to swap source 1 and source 2 |

---

## LFO (Low Frequency Oscillator)

Each LFO modulates a target parameter over time. Parameters are prefixed with the LFO's name.

| Parameter | Key Pattern | Min | Max | Default | Description |
|-----------|-------------|-----|-----|---------|-------------|
| Shape | `{name}_shape` | — | — | varies | Waveform shape (sine, triangle, square, sawtooth, noise, etc.) |
| Frequency | `{name}_frequency` | 0 | 1 | varies | Oscillation frequency (normalized; 1 = once per second) |
| Amplitude | `{name}_amplitude` | varies | varies | varies | Peak deviation from center applied to the target parameter |
| Phase | `{name}_phase` | 0 | 360 | varies | Starting phase offset of the waveform (degrees) |
| Seed | `{name}_seed` | 0 | 100 | varies | Random seed for noise-based waveforms |
| Noise Octaves | `{name}_noise_octaves` | 1 | 10 | 6 | Noise complexity; more octaves = more detail |
| Noise Persistence | `{name}_noise_persistence` | 0.1 | 1.0 | 0.5 | Amplitude falloff per octave in noise LFO |
| Noise Lacunarity | `{name}_noise_lacunarity` | 1.0 | 2.0 | 2.0 | Frequency multiplier per octave in noise LFO |
| Noise Repeat | `{name}_noise_repeat` | 1 | 1000 | 100 | Period at which the noise pattern loops |
| Noise Base | `{name}_noise_base` | 0 | 1000 | 456 | Offset into the noise field; changes the pattern shape |
| Cutoff Min | `{name}_cutoff_min` | varies | varies | min | Clamps the LFO output to a minimum value |
| Cutoff Max | `{name}_cutoff_max` | varies | varies | max | Clamps the LFO output to a maximum value |

---

## Audio Reactive

Each audio-reactive binding maps a frequency band to a target parameter. Parameters are prefixed with the binding's name.

| Parameter | Key Pattern | Min | Max | Default | Description |
|-----------|-------------|-----|-----|---------|-------------|
| Band | `{name}_band` | — | — | varies | Frequency band to listen to (sub, bass, mid, high, etc.) |
| Sensitivity | `{name}_sensitivity` | 0.0 | 5.0 | 1.0 | Gain applied to the audio signal before mapping to the parameter |
| Attack | `{name}_attack` | 0.0 | 1.0 | 0.3 | How quickly the output rises in response to a transient |
| Decay | `{name}_decay` | 0.0 | 1.0 | 0.1 | How quickly the output falls after the signal drops |
| Cutoff Min | `{name}_cutoff_min` | -100 | 100 | -100 | Minimum value the audio binding will drive the target to |
| Cutoff Max | `{name}_cutoff_max` | -100 | 100 | 100 | Maximum value the audio binding will drive the target to |
