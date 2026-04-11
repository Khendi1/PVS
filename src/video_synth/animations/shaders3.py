"""
GPU shader collection 3: generative systems, wave physics, and simulations.

Presets
-------
  Stateless  : WAVE_INTERFERENCE, LISSAJOUS_GLOW, SYMMETRY_BLOOM,
               FLOW_STREAMLINES, MAGNETIC_TERRAIN, HYPNODROME, SPECTRAL_CAUSTICS
  Stateful   : GRAY_SCOTT  (ping-pong reaction-diffusion)

Shared controls
---------------
  s3_speed       – time multiplier
  s3_zoom        – spatial scale
  s3_brightness  – output gain
  s3_color_shift – hue rotation (0-1)
  s3_evolution   – intrinsic drift rate (patterns slowly self-modulate)
  s3_symmetry    – N-fold rotational fold (1 = off, 2-12)
  s3_morph       – blend between two internal pattern states
  s3_palette     – colour palette dropdown
  s3_distortion  – warp / spread intensity
  s3_param_a/b/c – shader-specific knobs (labelled in 'info' strings)

Gray-Scott controls
-------------------
  s3_feed_rate  – feed rate  (morphology selector: spots / mazes / worms)
  s3_kill_rate  – kill rate  (morphology selector)
  s3_diff_u     – activator diffusion coefficient
  s3_diff_v     – inhibitor diffusion coefficient
  s3_sim_steps  – update passes per rendered frame (speed vs quality)

Requires: moderngl, numpy, opencv-python
"""

import cv2
import numpy as np
import time
import moderngl
import logging

from animations.base import Animation
from animations.enums import Shader3Type, Shader3Palette
from common import Widget

log = logging.getLogger(__name__)

STATEFUL_SHADERS = {Shader3Type.GRAY_SCOTT}

# ---------------------------------------------------------------------------
# Shared GLSL header injected into every fragment shader
# ---------------------------------------------------------------------------
_HEADER = """
#version 330
uniform vec2  u_resolution;
uniform float u_time;
uniform float u_zoom;
uniform float u_brightness;
uniform float u_color_shift;
uniform float u_evolution;
uniform float u_symmetry;
uniform float u_morph;
uniform float u_distortion;
uniform float u_param_a;
uniform float u_param_b;
uniform float u_param_c;
uniform float u_palette;
out vec4 f_color;

// ---- hash / noise --------------------------------------------------------
float hash11(float p) {
    p = fract(p * .1031); p *= p + 33.33; p *= p + p; return fract(p);
}
float hash21(vec2 p) {
    p = fract(p * vec2(234.34, 435.345));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
}
vec2 hash22(vec2 p) {
    vec3 a = fract(p.xyx * vec3(234.34, 435.345, 301.81));
    a += dot(a, a + 34.23);
    return fract(vec2(a.x * a.y, a.y * a.z));
}
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i),          hash21(i+vec2(1,0)), f.x),
               mix(hash21(i+vec2(0,1)),hash21(i+vec2(1,1)), f.x), f.y);
}
float fbm(vec2 p, int oct) {
    float v = 0.0, a = 0.5;
    mat2 R = mat2(0.8, 0.6, -0.6, 0.8);
    for (int i = 0; i < oct; i++) { v += a * noise(p); p = R * p * 2.1; a *= 0.5; }
    return v;
}

// ---- N-fold rotational symmetry -----------------------------------------
vec2 sym_fold(vec2 uv) {
    float sym = floor(u_symmetry);
    if (sym <= 1.0) return uv;
    float ang = atan(uv.y, uv.x);
    float r   = length(uv);
    float seg = 6.28318 / sym;
    ang = mod(ang, seg);
    ang = abs(ang - seg * 0.5);
    return vec2(cos(ang), sin(ang)) * r;
}

// ---- Cosine colour palette (7 modes) ------------------------------------
vec3 get_palette(float t) {
    t = fract(t * 0.5 + u_color_shift);
    vec3 a, b, c, d;
    int m = int(u_palette);
    if      (m == 0) { a=vec3(.5,.10,.02); b=vec3(.5,.30,.08); c=vec3(1.,.8,.3);  d=vec3(.0,.05,.1);  } // Fire
    else if (m == 1) { a=vec3(.15,.3,.55); b=vec3(.2,.3,.4);   c=vec3(.9,1.1,1.2);d=vec3(.0,.1,.25);  } // Ice
    else if (m == 2) { a=vec3(.5,.5,.5);   b=vec3(.5,.5,.5);   c=vec3(1.,1.,1.);  d=vec3(.0,.333,.667);} // Psychedelic
    else if (m == 3) { a=vec3(.04,.0,.08); b=vec3(.4,.3,.5);   c=vec3(1.,.5,1.);  d=vec3(.0,.25,.5);  } // Neon
    else if (m == 4) { a=vec3(.30,.2,.08); b=vec3(.2,.15,.05); c=vec3(.7,.5,.2);  d=vec3(.05,.15,.0); } // Earth
    else if (m == 5) { a=vec3(.02,.0,.04); b=vec3(.25,.1,.4);  c=vec3(.5,.3,1.);  d=vec3(.3,.5,.7);   } // Void
    else             { a=vec3(.5,.5,.5);   b=vec3(.5,.5,.5);   c=vec3(1.,.7,.4);  d=vec3(.0,.15,.3);  } // Spectrum
    return max(vec3(0.0), a + b * cos(6.28318 * (c * t + d)));
}
"""

# Display header for stateful shaders (adds u_state sampler)
_DISPLAY_HEADER = """
#version 330
uniform sampler2D u_state;
uniform vec2  u_resolution;
uniform float u_time;
uniform float u_brightness;
uniform float u_color_shift;
uniform float u_palette;
out vec4 f_color;

vec3 get_palette(float t) {
    t = fract(t * 0.5 + u_color_shift);
    vec3 a, b, c, d;
    int m = int(u_palette);
    if      (m == 0) { a=vec3(.5,.10,.02); b=vec3(.5,.30,.08); c=vec3(1.,.8,.3);  d=vec3(.0,.05,.1);  }
    else if (m == 1) { a=vec3(.15,.3,.55); b=vec3(.2,.3,.4);   c=vec3(.9,1.1,1.2);d=vec3(.0,.1,.25);  }
    else if (m == 2) { a=vec3(.5,.5,.5);   b=vec3(.5,.5,.5);   c=vec3(1.,1.,1.);  d=vec3(.0,.333,.667);}
    else if (m == 3) { a=vec3(.04,.0,.08); b=vec3(.4,.3,.5);   c=vec3(1.,.5,1.);  d=vec3(.0,.25,.5);  }
    else if (m == 4) { a=vec3(.30,.2,.08); b=vec3(.2,.15,.05); c=vec3(.7,.5,.2);  d=vec3(.05,.15,.0); }
    else if (m == 5) { a=vec3(.02,.0,.04); b=vec3(.25,.1,.4);  c=vec3(.5,.3,1.);  d=vec3(.3,.5,.7);   }
    else             { a=vec3(.5,.5,.5);   b=vec3(.5,.5,.5);   c=vec3(1.,.7,.4);  d=vec3(.0,.15,.3);  }
    return max(vec3(0.0), a + b * cos(6.28318 * (c * t + d)));
}
"""


class Shaders3(Animation):
    """
    Eight GPU-rendered generative presets with rich per-preset controls.

    All presets share speed / zoom / brightness / palette / evolution / symmetry.
    param_a, param_b, param_c are re-used across presets with semantic labels
    described in the 'info' field visible in the UI.

    GRAY_SCOTT is a true ping-pong reaction-diffusion simulation that
    accumulates state across frames.
    """

    def __init__(self, params, width=1280, height=720, group=None):
        super().__init__(params, width, height, group=group)
        subgroup = self.__class__.__name__
        self.ctx = moderngl.create_context(standalone=True)
        self.start_time = time.time()

        # ---- Shared parameters -------------------------------------------
        self.current_shader = params.new(
            "s3_type", min=0, max=len(Shader3Type) - 1, default=0,
            group=group, subgroup=subgroup,
            type=Widget.DROPDOWN, options=Shader3Type,
            info="Shader preset")

        self.speed = params.new(
            "s3_speed", min=0.0, max=3.0, default=1.0,
            subgroup=subgroup, group=group, info="Time speed")

        self.zoom = params.new(
            "s3_zoom", min=0.1, max=8.0, default=1.0,
            subgroup=subgroup, group=group, info="Spatial scale")

        self.brightness = params.new(
            "s3_brightness", min=0.0, max=3.0, default=1.0,
            subgroup=subgroup, group=group, info="Output brightness")

        self.color_shift = params.new(
            "s3_color_shift", min=0.0, max=1.0, default=0.0,
            subgroup=subgroup, group=group, info="Hue rotation")

        self.evolution = params.new(
            "s3_evolution", min=0.0, max=2.0, default=0.3,
            subgroup=subgroup, group=group,
            info="Intrinsic pattern drift rate")

        self.symmetry = params.new(
            "s3_symmetry", min=1, max=12, default=1,
            subgroup=subgroup, group=group,
            info="N-fold rotational symmetry (1 = off)")

        self.morph = params.new(
            "s3_morph", min=0.0, max=1.0, default=0.0,
            subgroup=subgroup, group=group,
            info="Blend between internal states")

        self.palette_mode = params.new(
            "s3_palette", min=0, max=len(Shader3Palette) - 1, default=2,
            group=group, subgroup=subgroup,
            type=Widget.DROPDOWN, options=Shader3Palette,
            info="Colour palette")

        self.distortion = params.new(
            "s3_distortion", min=0.0, max=2.0, default=0.5,
            subgroup=subgroup, group=group, info="Warp / spread intensity")

        self.param_a = params.new(
            "s3_param_a", min=0.0, max=10.0, default=3.0,
            subgroup=subgroup, group=group,
            info="A: sources / fx-ratio / dipoles / R-r ratio")

        self.param_b = params.new(
            "s3_param_b", min=0.0, max=10.0, default=2.0,
            subgroup=subgroup, group=group,
            info="B: wave-k / fy-ratio / falloff / d-ratio")

        self.param_c = params.new(
            "s3_param_c", min=0.0, max=10.0, default=1.0,
            subgroup=subgroup, group=group,
            info="C: extra variation / layer count")

        # ---- Gray-Scott specific -----------------------------------------
        self.feed_rate = params.new(
            "s3_feed_rate", min=0.010, max=0.080, default=0.037,
            subgroup=subgroup, group=group,
            info="GS feed rate  (spots ↔ mazes ↔ worms)")

        self.kill_rate = params.new(
            "s3_kill_rate", min=0.040, max=0.075, default=0.060,
            subgroup=subgroup, group=group,
            info="GS kill rate  (morphology selector)")

        self.diff_u = params.new(
            "s3_diff_u", min=0.05, max=0.30, default=0.2097,
            subgroup=subgroup, group=group,
            info="GS activator diffusion")

        self.diff_v = params.new(
            "s3_diff_v", min=0.02, max=0.15, default=0.1050,
            subgroup=subgroup, group=group,
            info="GS inhibitor diffusion")

        self.sim_steps = params.new(
            "s3_sim_steps", min=1, max=10, default=4,
            subgroup=subgroup, group=group,
            info="GS update steps per frame (speed vs quality)")

        # ---- Build GL resources ------------------------------------------
        self._build_programs()

        self.out_texture = self.ctx.texture((width, height), 3)
        self.out_fbo = self.ctx.framebuffer(color_attachments=[self.out_texture])

        # Ping-pong state textures for stateful shaders (float RGBA)
        self.sim_data: dict = {}
        for stype in STATEFUL_SHADERS:
            textures, fbos = [], []
            for _ in range(2):
                tex = self.ctx.texture((width, height), 4, dtype='f4')
                tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
                tex.repeat_x = True
                tex.repeat_y = True
                textures.append(tex)
                fbos.append(self.ctx.framebuffer(color_attachments=[tex]))
            self.sim_data[stype] = {
                'textures': textures, 'fbos': fbos,
                'index': 0, 'initialized': False,
            }

    # -----------------------------------------------------------------------
    # Program compilation
    # -----------------------------------------------------------------------
    def _build_programs(self):
        vert = """
            #version 330
            in vec2 in_vert;
            void main() { gl_Position = vec4(in_vert, 0.0, 1.0); }
        """
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4')
        self._vbo = self.ctx.buffer(quad)

        self.programs: dict = {}

        for stype, frag in self._stateless_frags().items():
            try:
                prog = self.ctx.program(vertex_shader=vert, fragment_shader=frag)
                vao = self.ctx.vertex_array(prog, [(self._vbo, '2f', 'in_vert')])
                self.programs[stype] = {'prog': prog, 'vao': vao}
            except Exception as e:
                log.error(f"Shaders3 compile [{stype.name}]: {e}")

        for stype, (upd_frag, disp_frag) in self._stateful_frags().items():
            try:
                up_p = self.ctx.program(vertex_shader=vert, fragment_shader=upd_frag)
                up_v = self.ctx.vertex_array(up_p, [(self._vbo, '2f', 'in_vert')])
                di_p = self.ctx.program(vertex_shader=vert, fragment_shader=disp_frag)
                di_v = self.ctx.vertex_array(di_p, [(self._vbo, '2f', 'in_vert')])
                self.programs[stype] = {
                    'update_prog': up_p, 'update_vao': up_v,
                    'display_prog': di_p, 'display_vao': di_v,
                }
            except Exception as e:
                log.error(f"Shaders3 compile [{stype.name}]: {e}")

    # -----------------------------------------------------------------------
    # Rendering helpers
    # -----------------------------------------------------------------------
    def _uniforms(self) -> dict:
        t = (time.time() - self.start_time) * float(self.speed.value)
        evo = (time.time() - self.start_time) * float(self.evolution.value) * 0.05
        return {
            'u_resolution':  (float(self.width), float(self.height)),
            'u_time':        t,
            'u_zoom':        float(self.zoom.value),
            'u_brightness':  float(self.brightness.value),
            'u_color_shift': float(self.color_shift.value),
            'u_evolution':   evo,
            'u_symmetry':    float(max(1, int(self.symmetry.value))),
            'u_morph':       float(self.morph.value),
            'u_distortion':  float(self.distortion.value),
            'u_param_a':     float(self.param_a.value),
            'u_param_b':     float(self.param_b.value),
            'u_param_c':     float(self.param_c.value),
            'u_palette':     float(int(self.palette_mode.value)),
            'u_feed_rate':   float(self.feed_rate.value),
            'u_kill_rate':   float(self.kill_rate.value),
            'u_diff_u':      float(self.diff_u.value),
            'u_diff_v':      float(self.diff_v.value),
        }

    @staticmethod
    def _set_uniforms(prog, uniforms: dict):
        for k, v in uniforms.items():
            if k in prog:
                prog[k].value = v

    def _read_fbo(self) -> np.ndarray:
        data = np.frombuffer(self.out_fbo.read(components=3), dtype=np.uint8)
        return cv2.flip(data.reshape((self.height, self.width, 3)), 0)

    def _render_stateless(self, stype: Shader3Type, uniforms: dict) -> np.ndarray:
        if stype not in self.programs:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        prog = self.programs[stype]['prog']
        vao  = self.programs[stype]['vao']
        self.out_fbo.use()
        self.ctx.clear()
        self._set_uniforms(prog, uniforms)
        vao.render(moderngl.TRIANGLE_STRIP)
        return self._read_fbo()

    # ---- Gray-Scott initialization ----------------------------------------
    def _init_gray_scott(self):
        stype = Shader3Type.GRAY_SCOTT
        sim   = self.sim_data[stype]
        w, h  = self.width, self.height
        state = np.zeros((h, w, 4), dtype=np.float32)
        state[:, :, 0] = 1.0          # U = 1 everywhere
        rng = np.random.default_rng()
        n_seeds = max(8, w * h // 1200)
        for _ in range(n_seeds):
            x = int(rng.integers(12, w - 12))
            y = int(rng.integers(12, h - 12))
            r = int(rng.integers(5, 14))
            y0, y1 = max(0, y - r), min(h, y + r)
            x0, x1 = max(0, x - r), min(w, x + r)
            state[y0:y1, x0:x1, 0] = 0.50
            state[y0:y1, x0:x1, 1] = 0.25
        raw = state.tobytes()
        for tex in sim['textures']:
            tex.write(raw)
        sim['index'] = 0
        sim['initialized'] = True

    def _render_stateful(self, stype: Shader3Type, uniforms: dict) -> np.ndarray:
        if stype not in self.programs:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        sim = self.sim_data[stype]
        if not sim['initialized']:
            if stype == Shader3Type.GRAY_SCOTT:
                self._init_gray_scott()

        up_prog = self.programs[stype]['update_prog']
        up_vao  = self.programs[stype]['update_vao']
        di_prog = self.programs[stype]['display_prog']
        di_vao  = self.programs[stype]['display_vao']

        # --- Update passes (ping-pong) ------------------------------------
        steps = max(1, int(self.sim_steps.value))
        for _ in range(steps):
            src = sim['index']
            dst = 1 - src
            sim['textures'][src].use(location=0)
            sim['fbos'][dst].use()
            self.ctx.clear()
            if 'u_state' in up_prog:
                up_prog['u_state'].value = 0
            self._set_uniforms(up_prog, uniforms)
            up_vao.render(moderngl.TRIANGLE_STRIP)
            sim['index'] = dst

        # --- Display pass -------------------------------------------------
        sim['textures'][sim['index']].use(location=0)
        self.out_fbo.use()
        self.ctx.clear()
        if 'u_state' in di_prog:
            di_prog['u_state'].value = 0
        self._set_uniforms(di_prog, uniforms)
        di_vao.render(moderngl.TRIANGLE_STRIP)
        return self._read_fbo()

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------
    def get_frame(self, frame: np.ndarray = None) -> np.ndarray:
        stype    = Shader3Type(int(self.current_shader.value))
        uniforms = self._uniforms()
        if stype in STATEFUL_SHADERS:
            return self._render_stateful(stype, uniforms)
        return self._render_stateless(stype, uniforms)

    # =======================================================================
    # GLSL source
    # =======================================================================

    def _stateless_frags(self) -> dict:
        """Return {Shader3Type: frag_src} for every stateless preset."""

        # ------------------------------------------------------------------
        # WAVE_INTERFERENCE
        # Superposition of circular waves emitted by moving point sources.
        # Sources trace Lissajous paths; frequency and phase are controllable.
        # param_a = number of sources (1-8)
        # param_b = wave number (spatial frequency)
        # distortion = source orbit radius
        # ------------------------------------------------------------------
        wave_interference = _HEADER + """
        void main() {
            vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom;
            float t  = u_time;
            uv = sym_fold(uv);

            int   n_src = max(1, min(8, int(u_param_a)));
            float k     = 2.0 + u_param_b * 4.0;   // wave number
            float r_orb = 0.1 + u_distortion * 0.4; // orbit radius
            float wave  = 0.0;

            for (int i = 0; i < n_src; i++) {
                float fi     = float(i);
                float fi_n   = fi / float(n_src);
                // Lissajous source path with slow evolution drift
                float fx = 1.0 + fi * 0.37 + sin(u_evolution * 0.3 + fi) * 0.12;
                float fy = 1.0 + fi * 0.53 + cos(u_evolution * 0.21 + fi) * 0.09;
                vec2 src = vec2(cos(t * fx + fi * 2.094) * r_orb,
                                sin(t * fy + fi * 2.094) * r_orb);
                float d   = length(uv - src);
                float phi = fi_n * 6.28318;           // phase offset per source
                // Morph: blend between constructive and destructive arrangements
                wave += sin(d * k - t * 2.0 + phi * (1.0 + u_morph));
            }
            wave /= float(n_src);
            float intensity = wave * 0.5 + 0.5;

            // Standing-wave contour lines
            float lines = smoothstep(0.03, 0.0, abs(fract(intensity * u_param_c * 4.0) - 0.5) - 0.1);

            vec3 col = get_palette(intensity + t * 0.04);
            col = mix(col, vec3(1.0), lines * 0.6);
            f_color = vec4(col * u_brightness, 1.0);
        }
        """

        # ------------------------------------------------------------------
        # LISSAJOUS_GLOW
        # Phosphor-accumulation rendering of Lissajous / Bowditch curves.
        # Frequency ratio slowly drifts with evolution, tracing new figures.
        # param_a = X frequency (fx)
        # param_b = Y frequency (fy)
        # distortion = phase between axes (0-pi)
        # morph = blend second curve with integer-snapped ratio
        # ------------------------------------------------------------------
        lissajous_glow = _HEADER + """
        void main() {
            vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom;
            float t = u_time * 0.5;
            uv = sym_fold(uv);

            // Primary frequency ratio (drifts with evolution)
            float fx = max(0.5, u_param_a + sin(u_evolution * 0.4) * u_distortion * 0.6);
            float fy = max(0.5, u_param_b + cos(u_evolution * 0.31) * u_distortion * 0.5);
            // Morphs toward the nearest rational ratio
            float fx2 = round(fx), fy2 = round(fy);
            fx = mix(fx, fx2, u_morph);
            fy = mix(fy, fy2, u_morph);

            float phase = u_distortion * 1.5707; // phase offset (0 – π/2)

            // Phosphor accumulation: sample many points on the curve
            float glow = 0.0;
            const int SAMPLES = 300;
            float period = 6.28318 * max(fx, fy);  // one full closed period
            for (int i = 0; i < SAMPLES; i++) {
                float s   = float(i) / float(SAMPLES) * period;
                vec2  p   = vec2(cos(fx * s + phase), sin(fy * s)) * 0.85;
                float d   = length(uv - p);
                glow += exp(-d * d * 60.0);
            }
            glow /= float(SAMPLES);

            float secondary = 0.0;
            float fx3 = fx + sin(t * 0.07) * 0.1;
            float fy3 = fy + cos(t * 0.05) * 0.1;
            for (int i = 0; i < 150; i++) {
                float s = float(i) / 150.0 * period;
                vec2  p = vec2(cos(fx3 * s + phase * 0.5), sin(fy3 * s)) * 0.85;
                secondary += exp(-length(uv - p) * length(uv - p) * 80.0);
            }
            secondary /= 150.0;

            vec3 col = get_palette(glow * 3.0 + t * 0.03);
            col     += get_palette(secondary * 3.0 + 0.5 + t * 0.03) * 0.4;
            col     *= (1.0 + glow * 3.0 + secondary);     // bloom
            f_color  = vec4(col * u_brightness, 1.0);
        }
        """

        # ------------------------------------------------------------------
        # SYMMETRY_BLOOM
        # N-fold rotational symmetry applied to an FBM domain-warp field.
        # The warp slowly self-modulates via the evolution parameter.
        # symmetry = fold count
        # param_a  = first warp layer strength
        # param_b  = FBM scale
        # distortion = second warp layer strength
        # ------------------------------------------------------------------
        symmetry_bloom = _HEADER + """
        void main() {
            vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom;
            float t  = u_time * 0.35;

            // Apply symmetry in world space
            uv = sym_fold(uv);

            // Slowly drifting offsets (evolution)
            float evo_ang = u_evolution * 0.7;
            vec2  drift   = vec2(cos(evo_ang), sin(evo_ang)) * 0.15;

            // Two-layer FBM domain warp
            vec2 q = vec2(fbm(uv * u_param_b + drift + t * 0.08, 4),
                          fbm(uv * u_param_b + drift + vec2(3.1, 7.3) + t * 0.06, 4));
            vec2 r = vec2(fbm(uv * u_param_b + u_param_a * q + t * 0.05, 5),
                          fbm(uv * u_param_b + u_param_a * q + vec2(9.2, 2.8) + t * 0.04, 5));

            float f = fbm(uv * u_param_b + u_distortion * r + t * 0.03, 6);

            // Colour layers
            vec3 col = mix(get_palette(f + t * 0.02),
                           get_palette(length(q) + 0.5 + t * 0.02),
                           clamp(length(r), 0.0, 1.0));
            col = mix(col, get_palette(f + length(r) * 0.6), u_morph);

            // Vignette
            float vig = 1.0 - dot(uv * 0.3, uv * 0.3);
            f_color = vec4(col * u_brightness * clamp(vig, 0.0, 1.0), 1.0);
        }
        """

        # ------------------------------------------------------------------
        # FLOW_STREAMLINES
        # Each pixel traces a streamline backward through a curl-noise field.
        # The field itself slowly evolves.  Different depths get different hues.
        # param_a = curl strength / step scale
        # param_b = noise spatial frequency
        # distortion = step length
        # morph = fade exponent (long vs short trails)
        # ------------------------------------------------------------------
        flow_streamlines = _HEADER + """
        void main() {
            vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom;
            float t = u_time * 0.25;
            uv = sym_fold(uv);

            vec3  col   = vec3(0.0);
            float total = 0.0;
            vec2  p     = uv;
            float freq  = 1.2 + u_param_b * 0.8;
            float step_l = 0.04 * (0.5 + u_distortion * 0.8);
            float fade_exp = 1.5 + u_morph * 2.5;

            const int STEPS = 18;
            for (int i = 0; i < STEPS; i++) {
                float fi = float(i) / float(STEPS);

                // Curl noise: perpendicular to gradient of fbm
                float eps = 0.007;
                float n0  = fbm(p * freq + t * 0.12, 4);
                float ndx = fbm((p + vec2(eps, 0.0)) * freq + t * 0.12, 4);
                float ndy = fbm((p + vec2(0.0, eps)) * freq + t * 0.12, 4);
                vec2  curl = vec2(ndy - n0, -(ndx - n0)) / eps * u_param_a;

                float w = exp(-fi * fade_exp);
                col    += get_palette(n0 + fi * 0.35 + t * 0.04) * w;
                total  += w;

                // Evolving secondary swirl driven by evolution parameter
                float swirl_ang = u_evolution * 0.5 + fi * 0.4;
                curl += vec2(cos(swirl_ang), sin(swirl_ang)) * 0.15;

                p += normalize(curl + vec2(1e-6)) * step_l;
            }
            col /= max(total, 1e-5);
            f_color = vec4(col * u_brightness, 1.0);
        }
        """

        # ------------------------------------------------------------------
        # MAGNETIC_TERRAIN
        # Scalar magnetic potential from multiple oscillating dipoles,
        # visualised as a height-map coloured with the palette.
        # param_a = number of dipoles (1-6)
        # param_b = field falloff (higher = shorter range)
        # distortion = pole oscillation eccentricity
        # morph = blend between field-lines and magnitude
        # ------------------------------------------------------------------
        magnetic_terrain = _HEADER + """
        void main() {
            vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom;
            float t = u_time;
            uv = sym_fold(uv);

            int n_poles = max(1, min(6, int(u_param_a)));
            float field_val = 0.0;
            vec2  grad      = vec2(0.0);

            for (int i = 0; i < n_poles; i++) {
                float fi    = float(i);
                float angle = fi / float(n_poles) * 6.28318;
                float orb   = 0.3 + u_distortion * 0.25;
                // Each dipole traces an ellipse that slowly drifts (evolution)
                float evo_phase = u_evolution * 0.8 + fi * 1.1;
                float ea  = orb * (1.0 + sin(evo_phase) * 0.25);
                float eb  = orb * (1.0 + cos(evo_phase) * 0.25);
                vec2 pole = vec2(cos(t * 0.4 + angle) * ea,
                                 sin(t * 0.35 + angle + 1.2) * eb);

                float sign  = (mod(fi, 2.0) < 1.0) ? 1.0 : -1.0;
                vec2  diff  = uv - pole;
                float dist2 = dot(diff, diff) + 0.01;
                float r_pow = pow(dist2, u_param_b * 0.5 + 0.5);
                field_val  += sign / r_pow;
                grad       += sign * (-2.0 * diff) / (r_pow * dist2);
            }

            float mag  = abs(field_val);
            float t_col = log(mag + 1.0) * 0.4;

            // Field line density as texture
            float line_t = fract(field_val * 1.5 + t * 0.1);
            float lines  = smoothstep(0.05, 0.0, abs(line_t - 0.5) - 0.1);

            vec3 col = mix(get_palette(t_col + t * 0.03),
                           get_palette(t_col + 0.5),
                           lines * u_morph);
            col += vec3(lines) * 0.3 * u_morph;

            f_color = vec4(col * u_brightness, 1.0);
        }
        """

        # ------------------------------------------------------------------
        # HYPNODROME
        # Superimposed hypotrochoid / epitrochoid (spirograph) curves.
        # The R/r ratio evolves slowly, sweeping through Lissajous-family figures.
        # param_a = R/r ratio (determines petal count)
        # param_b = d/r arm length ratio
        # distortion = phase spread between overlaid curves
        # morph = blend hypotrochoid ↔ epitrochoid
        # ------------------------------------------------------------------
        hypnodrome = _HEADER + """
        float hypo_dist(vec2 uv, float ratio, float d_ratio, float t_off) {
            // Trace hypotrochoid: r=1/ratio, d=d_ratio/ratio
            float r = 1.0 / max(ratio, 0.5);
            float d = d_ratio * r;
            float per = 6.28318 * max(ratio, 1.0);
            float glow = 0.0;
            const int S = 260;
            for (int i = 0; i < S; i++) {
                float s = float(i) / float(S) * per + t_off;
                vec2 p = vec2((1.0 - r) * cos(s) + d * cos((1.0 - r) / r * s),
                              (1.0 - r) * sin(s) - d * sin((1.0 - r) / r * s)) * 0.75;
                glow += exp(-dot(uv - p, uv - p) * 40.0);
            }
            return glow / float(S);
        }
        float epi_dist(vec2 uv, float ratio, float d_ratio, float t_off) {
            float r = 1.0 / max(ratio, 0.5);
            float d = d_ratio * r;
            float per = 6.28318 * max(ratio, 1.0);
            float glow = 0.0;
            const int S = 260;
            for (int i = 0; i < S; i++) {
                float s = float(i) / float(S) * per + t_off;
                vec2 p = vec2((1.0 + r) * cos(s) - d * cos((1.0 + r) / r * s),
                              (1.0 + r) * sin(s) - d * sin((1.0 + r) / r * s)) * 0.45;
                glow += exp(-dot(uv - p, uv - p) * 40.0);
            }
            return glow / float(S);
        }

        void main() {
            vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom;
            float t = u_time * 0.4;
            uv = sym_fold(uv);

            // Ratio drifts slowly with evolution
            float ratio   = max(1.5, u_param_a + sin(u_evolution * 0.25) * u_distortion * 0.8);
            float d_ratio = max(0.1, u_param_b * 0.5 + cos(u_evolution * 0.18) * u_distortion * 0.4);

            // Three overlapping curves with phase offset
            float g0 = mix(hypo_dist(uv, ratio,       d_ratio,       t),
                           epi_dist (uv, ratio,       d_ratio,       t),       u_morph);
            float g1 = mix(hypo_dist(uv, ratio + 0.5, d_ratio * 0.8, t + u_distortion),
                           epi_dist (uv, ratio + 0.5, d_ratio * 0.8, t + u_distortion), u_morph);
            float g2 = mix(hypo_dist(uv, ratio * 1.3, d_ratio * 1.2, t - u_distortion),
                           epi_dist (uv, ratio * 1.3, d_ratio * 1.2, t - u_distortion), u_morph);

            float combined = g0 + g1 * 0.5 + g2 * 0.3;
            vec3 col = get_palette(g0 * 4.0 + t * 0.05);
            col     += get_palette(g1 * 4.0 + 0.3 + t * 0.05) * 0.5;
            col     += get_palette(g2 * 4.0 + 0.6 + t * 0.05) * 0.3;
            col     *= (1.0 + combined * 2.5);       // phosphor bloom
            f_color  = vec4(col * u_brightness, 1.0);
        }
        """

        # ------------------------------------------------------------------
        # SPECTRAL_CAUSTICS
        # Light caustics via animated lens distortion applied per spectral band,
        # creating chromatic dispersion and shimmering highlights.
        # param_a = number of spectral bands / layers (1-8)
        # param_b = lens strength
        # distortion = wave perturbation amplitude
        # morph = blend between additive and saturated compositing
        # ------------------------------------------------------------------
        spectral_caustics = _HEADER + """
        void main() {
            vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom * 2.0;
            float t  = u_time * 0.35;
            uv = sym_fold(uv);

            int   n_bands = max(1, min(8, int(u_param_a)));
            vec3  col     = vec3(0.0);

            for (int i = 0; i < n_bands; i++) {
                float fi    = float(i) / float(n_bands);
                float phase = fi * 6.28318 + t * 0.12;

                // Each band has a slightly different lens
                float lens_k  = u_param_b * (0.4 + fi * 0.3);
                float evo_off = sin(u_evolution * 0.6 + fi * 1.4) * 0.2;

                // Noise-perturbed lens mapping
                vec2 perturb = vec2(
                    noise(uv * 2.0 + vec2(0.0, phase)) * u_distortion * 0.25,
                    noise(uv * 2.0 + vec2(phase, 0.0)) * u_distortion * 0.25
                );
                vec2 luv = uv + perturb + vec2(cos(phase + evo_off), sin(phase + evo_off)) * 0.05;

                // Caustic brightness: concentrated where the Jacobian is small
                float r = length(luv);
                float lens = exp(-r * lens_k) + exp(-r * r * lens_k * 0.5) * 0.4;
                float ripple = sin(r * (8.0 + u_param_b * 3.0) - t * 2.5 + fi) * 0.5 + 0.5;
                lens *= ripple * ripple;

                // Morph between three compositing modes
                col += get_palette(fi + t * 0.04 + length(perturb)) * lens;
            }
            col /= float(n_bands);
            // Saturate highlights
            col = mix(col, pow(col, vec3(0.5 + u_morph * 0.5)), u_morph);
            f_color = vec4(col * u_brightness, 1.0);
        }
        """

        return {
            Shader3Type.WAVE_INTERFERENCE: wave_interference,
            Shader3Type.LISSAJOUS_GLOW:    lissajous_glow,
            Shader3Type.SYMMETRY_BLOOM:    symmetry_bloom,
            Shader3Type.FLOW_STREAMLINES:  flow_streamlines,
            Shader3Type.MAGNETIC_TERRAIN:  magnetic_terrain,
            Shader3Type.HYPNODROME:        hypnodrome,
            Shader3Type.SPECTRAL_CAUSTICS: spectral_caustics,
        }

    def _stateful_frags(self) -> dict:
        """Return {Shader3Type: (update_frag, display_frag)} for stateful presets."""

        # ------------------------------------------------------------------
        # GRAY_SCOTT  (ping-pong reaction-diffusion)
        # R channel = U (activator), G channel = V (inhibitor).
        # Morphology guide (f = feed, k = kill):
        #   spots          f≈0.014  k≈0.054
        #   mitosis        f≈0.028  k≈0.062
        #   coral / mazes  f≈0.037  k≈0.060
        #   solitons       f≈0.022  k≈0.059
        #   worms          f≈0.046  k≈0.063
        #   spirals        f≈0.018  k≈0.051
        #
        # param_a = perturbation injected each frame (keeps alive)
        # param_b = colour contrast
        # param_c = background brightness
        # ------------------------------------------------------------------
        gs_update = """
        #version 330
        uniform sampler2D u_state;
        uniform vec2  u_resolution;
        uniform float u_feed_rate;
        uniform float u_kill_rate;
        uniform float u_diff_u;
        uniform float u_diff_v;
        uniform float u_param_a;  // micro-perturbation strength
        uniform float u_time;
        out vec4 f_color;

        // Cheap hash for stochastic perturbation
        float hash12(vec2 p) {
            vec3 p3 = fract(vec3(p.xyx) * 0.1031);
            p3 += dot(p3, p3.yzx + 33.33);
            return fract((p3.x + p3.y) * p3.z);
        }

        void main() {
            vec2 texel = 1.0 / u_resolution;
            vec2 coord = gl_FragCoord.xy * texel;

            // 9-point stencil (weights: centre=-1, cardinal=0.2, diagonal=0.05)
            vec4 c  = texture(u_state, coord);
            vec4 cN = texture(u_state, coord + vec2( 0.0,  texel.y));
            vec4 cS = texture(u_state, coord + vec2( 0.0, -texel.y));
            vec4 cE = texture(u_state, coord + vec2( texel.x,  0.0));
            vec4 cW = texture(u_state, coord + vec2(-texel.x,  0.0));
            vec4 cNE= texture(u_state, coord + vec2( texel.x,  texel.y));
            vec4 cNW= texture(u_state, coord + vec2(-texel.x,  texel.y));
            vec4 cSE= texture(u_state, coord + vec2( texel.x, -texel.y));
            vec4 cSW= texture(u_state, coord + vec2(-texel.x, -texel.y));

            float lapU = -c.r
                + 0.20 * (cN.r + cS.r + cE.r + cW.r)
                + 0.05 * (cNE.r + cNW.r + cSE.r + cSW.r);
            float lapV = -c.g
                + 0.20 * (cN.g + cS.g + cE.g + cW.g)
                + 0.05 * (cNE.g + cNW.g + cSE.g + cSW.g);

            float U   = c.r;
            float V   = c.g;
            float UVV = U * V * V;

            float dU = u_diff_u * lapU - UVV + u_feed_rate * (1.0 - U);
            float dV = u_diff_v * lapV + UVV - (u_feed_rate + u_kill_rate) * V;

            // Tiny stochastic perturbation (keeps pattern evolving)
            float perturb = hash12(coord + fract(vec2(u_time * 0.01))) * u_param_a * 0.002;
            dV += perturb;

            f_color = vec4(
                clamp(U + dU, 0.0, 1.0),
                clamp(V + dV, 0.0, 1.0),
                0.0, 1.0);
        }
        """

        gs_display = _DISPLAY_HEADER + """
        uniform float u_param_b;   // colour contrast
        uniform float u_param_c;   // background lift
        uniform float u_distortion; // hue offset

        void main() {
            vec2 coord = gl_FragCoord.xy / u_resolution;
            vec4 state = texture(u_state, coord);
            float U = state.r;
            float V = state.g;

            // V alone gives the cleanest morphology view
            float t_col = V * (1.0 + u_param_b * 2.0);

            // Optionally show U-V difference (morph between two views)
            // (u_param_c lifts the background)
            vec3 col = get_palette(t_col + u_time * 0.005 + u_distortion * 0.2);
            col      = mix(col * U + vec3(u_param_c * 0.05), col, clamp(V * 3.0, 0.0, 1.0));

            f_color = vec4(col * u_brightness, 1.0);
        }
        """

        return {
            Shader3Type.GRAY_SCOTT: (gs_update, gs_display),
        }
