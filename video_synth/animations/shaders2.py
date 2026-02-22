"""
Second GPU shader collection: abstract & generative visual programs.

Focuses on noise fields, signed distance functions, raymarching,
cellular patterns, and other generative techniques distinct from
the original Shaders class.

Requires: moderngl, numpy, opencv-python
"""

import cv2
import numpy as np
import time
import moderngl
import logging

from animations.base import Animation
from animations.enums import Shader2Type
from common import Widget

log = logging.getLogger(__name__)


class Shaders2(Animation):

    def __init__(self, params, width=1280, height=720, group=None):
        super().__init__(params, width, height, group=group)
        subgroup = self.__class__.__name__
        self.width, self.height = width, height
        self.ctx = moderngl.create_context(standalone=True)
        self.time = time.time()

        # --- Parameters ---
        self.current_shader = params.add("s2_type",
                                         min=0, max=len(Shader2Type) - 1, default=0,
                                         group=group, subgroup=subgroup,
                                         type=Widget.DROPDOWN, options=Shader2Type)
        self.zoom = params.add("s2_zoom",
                               min=0.1, max=10.0, default=1.0,
                               subgroup=subgroup, group=group)
        self.distortion = params.add("s2_distortion",
                                     min=0.0, max=2.0, default=0.5,
                                     subgroup=subgroup, group=group)
        self.iterations = params.add("s2_iterations",
                                     min=1.0, max=20.0, default=6.0,
                                     subgroup=subgroup, group=group)
        self.color_shift = params.add("s2_color_shift",
                                      min=0.0, max=6.28, default=0.0,
                                      subgroup=subgroup, group=group)
        self.brightness = params.add("s2_brightness",
                                     min=0.0, max=3.0, default=1.0,
                                     subgroup=subgroup, group=group)
        self.speed = params.add("s2_speed",
                                min=0.0, max=3.0, default=1.0,
                                subgroup=subgroup, group=group)
        self.param_a = params.add("s2_param_a",
                                  min=0.0, max=10.0, default=3.0,
                                  subgroup=subgroup, group=group)
        self.param_b = params.add("s2_param_b",
                                  min=0.0, max=10.0, default=2.0,
                                  subgroup=subgroup, group=group)

        # --- Build shader programs ---
        vertex_shader, code = self._get_shader_code()

        self.programs = {}
        vertices = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4')
        vbo = self.ctx.buffer(vertices)

        for name, frag_code in code.items():
            try:
                prog = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=frag_code)
                vao = self.ctx.vertex_array(prog, [(vbo, '2f', 'in_vert')])
                self.programs[name] = {'prog': prog, 'vao': vao}
            except Exception as e:
                log.error(f"Error compiling shader2 {name.name}: {e}")

        self.fbo_texture = self.ctx.texture((width, height), 3)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.fbo_texture])

    def render(self, params):
        if self.current_shader.value not in self.programs:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        prog = self.programs[self.current_shader.value]['prog']
        vao = self.programs[self.current_shader.value]['vao']

        for name, value in params.items():
            if name in prog:
                prog[name].value = value

        self.ctx.clear(0, 0, 0)
        self.fbo.use()
        vao.render(moderngl.TRIANGLE_STRIP)

        data = np.frombuffer(self.fbo.read(components=3), dtype=np.uint8)
        img = data.reshape((self.height, self.width, 3))
        return cv2.flip(img, 0)

    def get_frame(self, frame: np.ndarray = None):
        params = {
            'u_resolution': (float(self.width), float(self.height)),
            'u_time': (time.time() - self.time) * self.speed.value,
            'u_zoom': self.zoom.value,
            'u_distortion': self.distortion.value,
            'u_iterations': self.iterations.value,
            'u_color_shift': self.color_shift.value,
            'u_brightness': self.brightness.value,
            'u_param_a': self.param_a.value,
            'u_param_b': self.param_b.value,
        }
        return self.render(params)

    # -------------------------------------------------------------------------
    # Shader source code
    # -------------------------------------------------------------------------
    def _get_shader_code(self):
        vertex_shader = '''
            #version 330
            in vec2 in_vert;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
            }
        '''

        # Common GLSL header shared by all fragment shaders
        header = '''
            #version 330
            uniform vec2  u_resolution;
            uniform float u_time;
            uniform float u_zoom;
            uniform float u_distortion;
            uniform float u_iterations;
            uniform float u_color_shift;
            uniform float u_brightness;
            uniform float u_param_a;
            uniform float u_param_b;
            out vec4 f_color;

            // --- Shared utility functions ---
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
                vec2 i = floor(p);
                vec2 f = fract(p);
                f = f * f * (3.0 - 2.0 * f);
                float a = hash21(i);
                float b = hash21(i + vec2(1.0, 0.0));
                float c = hash21(i + vec2(0.0, 1.0));
                float d = hash21(i + vec2(1.0, 1.0));
                return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
            }

            float fbm(vec2 p, int octaves) {
                float v = 0.0, a = 0.5;
                mat2 rot = mat2(0.8, 0.6, -0.6, 0.8);
                for (int i = 0; i < octaves; i++) {
                    v += a * noise(p);
                    p = rot * p * 2.0;
                    a *= 0.5;
                }
                return v;
            }

            vec3 palette(float t) {
                vec3 a = vec3(0.5);
                vec3 b = vec3(0.5);
                vec3 c = vec3(1.0);
                vec3 d = vec3(0.0, 0.33, 0.67);
                return a + b * cos(6.28318 * (c * t + d + u_color_shift));
            }
        '''

        code = {

            # ---------------------------------------------------------------
            # DOMAIN WARP: FBM-warped FBM creating organic flowing structures
            # ---------------------------------------------------------------
            Shader2Type.DOMAIN_WARP: header + '''
                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom * 3.0;
                    float t = u_time * 0.3;

                    // First domain warp layer
                    vec2 q = vec2(
                        fbm(uv + vec2(0.0, 0.0) + t * 0.1, int(u_iterations)),
                        fbm(uv + vec2(5.2, 1.3) + t * 0.15, int(u_iterations))
                    );

                    // Second domain warp layer
                    vec2 r = vec2(
                        fbm(uv + u_param_a * q + vec2(1.7, 9.2) + t * 0.12, int(u_iterations)),
                        fbm(uv + u_param_a * q + vec2(8.3, 2.8) + t * 0.08, int(u_iterations))
                    );

                    float f = fbm(uv + u_param_b * r, int(u_iterations));

                    vec3 col = mix(vec3(0.1, 0.2, 0.4), vec3(0.9, 0.6, 0.2), clamp(f * f * 2.0, 0.0, 1.0));
                    col = mix(col, vec3(0.0, 0.2, 0.4), clamp(length(q), 0.0, 1.0));
                    col = mix(col, palette(f + length(r) * 0.5), clamp(length(r.x), 0.0, 1.0));

                    // Vignette
                    vec2 vuv = gl_FragCoord.xy / u_resolution - 0.5;
                    col *= 1.0 - dot(vuv, vuv) * u_distortion;

                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',

            # ---------------------------------------------------------------
            # VORONOI FLOW: Animated voronoi cells with flowing edges
            # ---------------------------------------------------------------
            Shader2Type.VORONOI_FLOW: header + '''
                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom * 4.0;
                    float t = u_time * 0.5;

                    float minDist = 1e10;
                    float secondMin = 1e10;
                    vec2 closestPoint = vec2(0.0);

                    // Voronoi with smooth animation
                    vec2 iuv = floor(uv);
                    vec2 fuv = fract(uv);

                    for (int y = -1; y <= 1; y++) {
                        for (int x = -1; x <= 1; x++) {
                            vec2 neighbor = vec2(float(x), float(y));
                            vec2 point = hash22(iuv + neighbor);
                            // Animate points
                            point = 0.5 + 0.5 * sin(t * u_param_a * 0.3 + 6.28 * point);
                            vec2 diff = neighbor + point - fuv;
                            float d = length(diff);
                            if (d < minDist) {
                                secondMin = minDist;
                                minDist = d;
                                closestPoint = iuv + neighbor + point;
                            } else if (d < secondMin) {
                                secondMin = d;
                            }
                        }
                    }

                    // Edge detection
                    float edge = secondMin - minDist;
                    float edgeLine = smoothstep(0.0, 0.05 + u_distortion * 0.1, edge);

                    // Color based on cell ID
                    float cellId = hash21(floor(closestPoint));
                    vec3 col = palette(cellId + t * 0.1);

                    // Darken edges
                    col *= edgeLine;

                    // Inner glow
                    col += palette(cellId + 0.5) * exp(-minDist * u_param_b) * 0.5;

                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',

            # ---------------------------------------------------------------
            # SDF MORPH: Morphing signed distance field shapes
            # ---------------------------------------------------------------
            Shader2Type.SDF_MORPH: header + '''
                float sdCircle(vec2 p, float r) { return length(p) - r; }
                float sdBox(vec2 p, vec2 b) {
                    vec2 d = abs(p) - b;
                    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
                }
                float sdTriangle(vec2 p, float r) {
                    const float k = sqrt(3.0);
                    p.x = abs(p.x) - r;
                    p.y = p.y + r / k;
                    if (p.x + k * p.y > 0.0) p = vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0;
                    p.x -= clamp(p.x, -2.0 * r, 0.0);
                    return -length(p) * sign(p.y);
                }
                float sdStar(vec2 p, float r, int n, float m) {
                    float an = 3.14159 / float(n);
                    float en = 3.14159 / m;
                    vec2 acs = vec2(cos(an), sin(an));
                    vec2 ecs = vec2(cos(en), sin(en));
                    float bn = mod(atan(p.x, p.y), 2.0 * an) - an;
                    p = length(p) * vec2(cos(bn), abs(sin(bn)));
                    p -= r * acs;
                    p += ecs * clamp(-dot(p, ecs), 0.0, r * acs.y / ecs.y);
                    return length(p) * sign(p.x);
                }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom * 2.0;
                    float t = u_time;
                    vec3 col = vec3(0.0);

                    for (float i = 0.0; i < u_iterations; i++) {
                        float phase = t * 0.3 + i * 1.5;
                        float morph = sin(phase) * 0.5 + 0.5;
                        float morph2 = sin(phase * 0.7 + 1.0) * 0.5 + 0.5;

                        // Morph between shapes
                        vec2 offset = vec2(
                            sin(t * 0.4 + i * 2.0) * u_distortion,
                            cos(t * 0.3 + i * 2.5) * u_distortion
                        );
                        vec2 p = uv + offset;

                        // Rotate per iteration
                        float a = t * 0.2 + i * 0.5;
                        mat2 rot = mat2(cos(a), -sin(a), sin(a), cos(a));
                        p *= rot;

                        float r = 0.2 + i * 0.05;
                        float d1 = sdCircle(p, r);
                        float d2 = sdBox(p, vec2(r * 0.8));
                        float d3 = sdTriangle(p, r);
                        float d4 = sdStar(p, r * 0.7, 5, u_param_a);

                        float d = mix(mix(d1, d2, morph), mix(d3, d4, morph2), sin(phase * 0.5) * 0.5 + 0.5);

                        // Glow
                        float glow = exp(-abs(d) * u_param_b * 3.0);
                        col += palette(i * 0.15 + t * 0.1 + d * 2.0) * glow * 0.6;

                        // Edge line
                        col += palette(i * 0.15 + 0.5) * smoothstep(0.02, 0.0, abs(d)) * 0.4;
                    }

                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',

            # ---------------------------------------------------------------
            # REACTION: GPU reaction-diffusion-like patterns
            # ---------------------------------------------------------------
            Shader2Type.REACTION: header + '''
                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom * 5.0;
                    float t = u_time * 0.4;

                    // Layered reaction-diffusion approximation via layered sine patterns
                    vec3 col = vec3(0.0);
                    float val = 0.0;

                    for (float i = 0.0; i < u_iterations; i++) {
                        float freq = u_param_a + i * 1.5;
                        float phase = t * (0.2 + i * 0.05);

                        vec2 p = uv;
                        // Domain rotation per layer
                        float a = i * 0.5 + t * 0.05;
                        mat2 rot = mat2(cos(a), -sin(a), sin(a), cos(a));
                        p = rot * p;

                        // Turing-like pattern: competing activator/inhibitor wavelengths
                        float activator = sin(p.x * freq + phase) * sin(p.y * freq * 1.1 + phase * 0.7);
                        float inhibitor = sin(p.x * freq * u_param_b + phase * 1.3) *
                                         sin(p.y * freq * u_param_b * 0.9 + phase * 0.9);

                        val += (activator - inhibitor * u_distortion) / (i + 1.0);
                    }

                    // Map to color
                    val = val * 0.5 + 0.5;
                    col = palette(val + t * 0.05);

                    // Contrast enhancement
                    col = pow(col, vec3(1.0 + u_distortion * 0.5));

                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',

            # ---------------------------------------------------------------
            # TUNNEL: Raymarched infinite tunnel
            # ---------------------------------------------------------------
            Shader2Type.TUNNEL: header + '''
                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y;
                    float t = u_time;

                    // Polar coordinates for tunnel
                    float angle = atan(uv.y, uv.x);
                    float radius = length(uv);

                    // Avoid division by zero at center
                    float depth = u_zoom * 0.5 / (radius + 0.001);

                    // Tunnel texture coordinates
                    float tx = angle / 3.14159 * u_param_a;
                    float ty = depth + t;

                    // Layered tunnel walls
                    vec3 col = vec3(0.0);
                    for (float i = 0.0; i < u_iterations; i++) {
                        float freq = 1.0 + i * 0.5;
                        float wall = sin(tx * freq * 3.0 + i) * sin(ty * freq + i * 0.5);
                        wall = smoothstep(0.0, u_distortion + 0.1, abs(wall));

                        col += palette(ty * 0.1 + i * 0.2) * wall / (i + 1.0);
                    }

                    // Depth fog
                    float fog = exp(-radius * u_param_b);
                    col *= fog;

                    // Center glow
                    col += palette(t * 0.1) * exp(-radius * 8.0) * 0.3;

                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',

            # ---------------------------------------------------------------
            # KALEIDO: Kaleidoscopic fractal with polar symmetry
            # ---------------------------------------------------------------
            Shader2Type.KALEIDO: header + '''
                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom;
                    float t = u_time * 0.5;

                    // Polar
                    float angle = atan(uv.y, uv.x);
                    float dist = length(uv);

                    // Kaleidoscope fold
                    float segments = floor(u_param_a + 2.0);
                    float segAngle = 6.28318 / segments;
                    angle = mod(angle, segAngle);
                    angle = abs(angle - segAngle * 0.5);

                    // Back to cartesian
                    uv = vec2(cos(angle), sin(angle)) * dist;

                    // Iterated fractal in the folded space
                    vec3 col = vec3(0.0);
                    vec2 z = uv;

                    for (float i = 0.0; i < u_iterations; i++) {
                        z = abs(z) / dot(z, z) - vec2(u_param_b * 0.1 + sin(t * 0.2) * u_distortion);

                        float d = length(z);
                        col += palette(d + i * 0.15 + t * 0.1) * exp(-d * 3.0) * 0.4;
                    }

                    // Vignette
                    col *= 1.0 - smoothstep(0.8, 2.0, dist);

                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',

            # ---------------------------------------------------------------
            # FLUID SIM: Approximated fluid dynamics via curl noise
            # ---------------------------------------------------------------
            Shader2Type.FLUID: header + '''
                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom * 3.0;
                    float t = u_time * 0.3;

                    vec3 col = vec3(0.0);
                    vec2 vel = vec2(0.0);

                    for (float i = 0.0; i < u_iterations; i++) {
                        float scale = 1.0 + i * 0.5;
                        vec2 p = uv * scale + vel;

                        // Curl noise approximation for velocity field
                        float eps = 0.01;
                        float n  = fbm(p + vec2(0.0, 0.0) + t * (0.1 + i * 0.02), 5);
                        float nx = fbm(p + vec2(eps, 0.0) + t * (0.1 + i * 0.02), 5);
                        float ny = fbm(p + vec2(0.0, eps) + t * (0.1 + i * 0.02), 5);

                        // Curl: perpendicular to gradient
                        vec2 curl = vec2(ny - n, -(nx - n)) / eps;
                        vel += curl * u_distortion * 0.02;

                        float density = abs(n - 0.5) * 2.0;
                        density = smoothstep(0.3, 0.0, density);

                        col += palette(n + i * 0.2 + t * 0.05) * density / (i * 0.5 + 1.0);
                    }

                    // Slight darkening at edges
                    vec2 vuv = gl_FragCoord.xy / u_resolution - 0.5;
                    col *= 1.0 - dot(vuv, vuv) * u_param_b * 0.5;

                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',

            # ---------------------------------------------------------------
            # ELECTRIC: Lightning / electric arc patterns
            # ---------------------------------------------------------------
            Shader2Type.ELECTRIC: header + '''
                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y;
                    float t = u_time;
                    vec3 col = vec3(0.0);

                    for (float i = 0.0; i < u_iterations; i++) {
                        float offset = i / u_iterations - 0.5;
                        vec2 p = uv;
                        p.y -= offset * u_zoom;

                        // Noisy line: x position varies with noise
                        float freq = u_param_a * (3.0 + i);
                        float n = 0.0;
                        float amp = 0.3 * u_distortion;
                        float f = freq;

                        for (int j = 0; j < 5; j++) {
                            n += amp * noise(vec2(p.x * f + t * (2.0 + i * 0.5), t * 0.5 + i * 10.0));
                            f *= 2.0;
                            amp *= 0.5;
                        }

                        float d = abs(p.y - n);

                        // Sharp glow falloff
                        float glow = exp(-d * u_param_b * 50.0);
                        float core = exp(-d * 300.0);

                        vec3 arcColor = palette(i * 0.3 + t * 0.1);
                        col += arcColor * glow * 0.5 + vec3(0.8, 0.9, 1.0) * core;
                    }

                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',

            # ---------------------------------------------------------------
            # CELLULAR: Smooth cellular / organic noise patterns
            # ---------------------------------------------------------------
            Shader2Type.CELLULAR: header + '''
                float worley(vec2 p) {
                    vec2 i = floor(p);
                    vec2 f = fract(p);
                    float minDist = 1.0;
                    for (int y = -1; y <= 1; y++) {
                        for (int x = -1; x <= 1; x++) {
                            vec2 neighbor = vec2(float(x), float(y));
                            vec2 point = hash22(i + neighbor);
                            point = 0.5 + 0.5 * sin(u_time * 0.5 + 6.28 * point);
                            float d = length(neighbor + point - f);
                            minDist = min(minDist, d);
                        }
                    }
                    return minDist;
                }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom * 5.0;
                    float t = u_time * 0.3;
                    vec3 col = vec3(0.0);

                    for (float i = 0.0; i < u_iterations; i++) {
                        float scale = u_param_a * (1.0 + i * 0.7);
                        vec2 p = uv * scale;

                        // Domain warp the worley input
                        p += vec2(
                            noise(uv * 2.0 + t + i) * u_distortion,
                            noise(uv * 2.0 + t + i + 100.0) * u_distortion
                        );

                        float w = worley(p);
                        float pattern = smoothstep(0.0, 0.5, w);

                        col += palette(w + i * 0.2 + t * 0.1) * (1.0 - pattern) / (i * 0.5 + 1.0);
                    }

                    // Subtle FBM overlay
                    float f = fbm(uv * u_param_b + t * 0.1, 4);
                    col += palette(f) * f * 0.15;

                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',

            # ---------------------------------------------------------------
            # WORMHOLE: Warped space tunnel with gravitational lensing
            # ---------------------------------------------------------------
            Shader2Type.WORMHOLE: header + '''
                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y;
                    float t = u_time * 0.5;

                    float dist = length(uv);
                    float angle = atan(uv.y, uv.x);

                    // Gravitational lens distortion
                    float warp = u_param_a * 0.1 / (dist + 0.1);
                    angle += warp * sin(t * 0.3);
                    dist = pow(dist, 1.0 + u_distortion * sin(t * 0.2));

                    // Accretion disk layers
                    vec3 col = vec3(0.0);
                    for (float i = 0.0; i < u_iterations; i++) {
                        float ringDist = abs(dist - 0.3 - i * 0.08);
                        float ring = exp(-ringDist * u_param_b * 20.0);

                        float tex = sin(angle * (5.0 + i * 2.0) + t * (1.0 + i * 0.3)) * 0.5 + 0.5;
                        tex *= sin(dist * 20.0 - t * 2.0 + i) * 0.5 + 0.5;

                        col += palette(dist + i * 0.15 + t * 0.1) * ring * tex;
                    }

                    // Event horizon
                    float horizon = smoothstep(0.15, 0.05, dist);
                    col = mix(col, vec3(0.0), horizon);

                    // Hawking radiation glow at the edge
                    col += palette(t * 0.2) * smoothstep(0.2, 0.1, dist) * (1.0 - horizon) * 0.5;

                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
        }

        return vertex_shader, code
