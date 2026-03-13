import cv2
import numpy as np
import time
import moderngl
import logging

from animations.base import Animation
from animations.enums import ShaderType
from common import Widget

log = logging.getLogger(__name__)

class Shaders(Animation):
    def __init__(self, params, width=1280, height=720, group=None):
        super().__init__(params, width, height, group=group)
        subgroup = self.__class__.__name__
        p_name = group.name.lower()
        self.width, self.height = width, height
        self.ctx = moderngl.create_context(standalone=True)
        self.time = time.time()

        self.current_shader = params.add("s_type",
                                         min=0, max=len(ShaderType)-1, default=0,
                                         group=group, subgroup=subgroup,
                                         type=Widget.DROPDOWN, options=ShaderType)
        self.zoom = params.add("s_zoom",
                               min=0.1, max=5.0, default=1.5,
                               subgroup=subgroup, group=group)
        self.distortion = params.add("s_distortion",
                                     min=0.0, max=1.0, default=0.5,
                                     subgroup=subgroup, group=group)
        self.iterations = params.add("s_iterations",
                                     min=1.0, max=10.0, default=4.0,
                                     subgroup=subgroup, group=group)
        self.color_shift = params.add("s_color_shift",
                                      min=0.5, max=3.0, default=1.0,
                                      subgroup=subgroup, group=group)
        self.brightness = params.add("s_brightness",
                                     min=0.0, max=2.0, default=1.0,
                                     subgroup=subgroup, group=group)
        self.hue_shift = params.add("s_hue_shift",
                                    min=0.0, max=7, default=0.0,
                                    subgroup=subgroup, group=group)
        self.saturation = params.add("s_saturation",
                                     min=0.0, max=2.0, default=1.0,
                                     subgroup=subgroup, group=group)
        self.x_shift = params.add("s_x_shift",
                                  min=-5.0, max=5.0, default=0.0,
                                  subgroup=subgroup, group=group)
        self.y_shift = params.add("s_y_shift",
                                  min=-5.0, max=5.0, default=0.0,
                                  subgroup=subgroup, group=group)
        self.rotation = params.add("s_rotation",
                                   min=-3.14, max=3.14, default=0.0,
                                   subgroup=subgroup, group=group)
        self.speed = params.add("s_speed",
                                min=0.0, max=2.0, default=1.0,
                                subgroup=subgroup, group=group)
        self.prev_shader = self.current_shader.value
        
        vertex_shader, code = self.get_shader_code()
        
        self.programs = {}
        vertices = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4')
        vbo = self.ctx.buffer(vertices)
        
        for name, frag_code in code.items():
            try:
                prog = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=frag_code)
                vao = self.ctx.vertex_array(prog, [(vbo, '2f', 'in_vert')])
                self.programs[name] = {'prog': prog, 'vao': vao}
            except Exception as e:
                log.error(f"Error compiling shader {name.name}: {e}")

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
        if self.current_shader.value != self.prev_shader:
            self.prev_shader = self.current_shader.value

        params = {
            'u_resolution': (self.width, self.height),
            'u_time': (time.time() - self.time) * self.speed.value,
            'u_zoom': self.zoom.value,
            'u_distortion': self.distortion.value,
            'u_iterations': self.iterations.value,
            'u_color_shift': self.color_shift.value,
            'u_brightness': self.brightness.value,
            'u_hue_shift': self.hue_shift.value,
            'u_saturation': self.saturation.value,
            'u_scroll_x': self.x_shift.value,
            'u_scroll_y': self.y_shift.value,
            'u_rotation': self.rotation.value,
        }
        return self.render(params)

    def get_shader_code(self):
        vertex_shader = '''
            #version 330
            in vec2 in_vert;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
            }
        '''
        code = {
            ShaderType.FRACTAL_0: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_color_shift;
                uniform float u_brightness;
                uniform float u_hue_shift;
                uniform float u_saturation;
                uniform float u_scroll_x;
                uniform float u_scroll_y;
                uniform float u_rotation;

                out vec4 f_color;

                vec3 palette(float t) {
                    vec3 a = vec3(0.5, 0.5, 0.5);
                    vec3 b = vec3(0.5, 0.5, 0.5);
                    vec3 c = vec3(1.0, 1.0, 1.0);
                    vec3 d = vec3(0.263, 0.416, 0.557);
                    return a + b * cos(6.28318 * (c * t * u_color_shift + d + u_hue_shift));
                }

                mat2 rotate2d(float angle){
                    return mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
                }

                void main() {
                    vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution.xy) / u_resolution.y;
                    
                    // Apply rotation
                    uv *= rotate2d(u_rotation);
                    
                    // Apply scrolling
                    uv += vec2(u_scroll_x, u_scroll_y);
                    
                    vec2 uv0 = uv;
                    vec3 finalColor = vec3(0.0);
                    
                    for (float i = 0.0; i < u_iterations; i++) {
                        uv = fract(uv * u_zoom) - 0.5;
                        float d = length(uv) * exp(-length(uv0));
                        vec3 col = palette(length(uv0) + i * 0.4 + u_time * 0.4);
                        d = sin(d * 8.0 + u_time) / 8.0;
                        d = abs(d);
                        d = pow(0.01 / d, 1.2 + u_distortion * 0.5);
                        finalColor += col * d;
                    }
                    
                    // Apply saturation
                    float gray = dot(finalColor, vec3(0.299, 0.587, 0.114));
                    finalColor = mix(vec3(gray), finalColor, u_saturation);
                    
                    f_color = vec4(finalColor * u_brightness, 1.0);
                }''',
            ShaderType.FRACTAL: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_color_shift;
                uniform float u_brightness;
                out vec4 f_color;

                vec3 palette(float t) {
                    return 0.5 + 0.5 * cos(6.28 * (t * u_color_shift + vec3(0.0, 0.33, 0.67)));
                }

                void main() {
                    vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution) / u_resolution.y;
                    vec2 uv0 = uv;
                    vec3 col = vec3(0.0);
                    
                    for (float i = 0.0; i < u_iterations; i++) {
                        uv = fract(uv * u_zoom) - 0.5;
                        float d = length(uv) * exp(-length(uv0));
                        col += palette(length(uv0) + i * 0.4 + u_time * 0.4) * (0.01 / abs(d));
                    }
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            ShaderType.GRID: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                out vec4 f_color;

                mat2 rot(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom * 3.0;
                    vec3 col = vec3(0.0);
                    
                    for(float i = 0.0; i < u_iterations; i++){
                        vec2 gv = fract(uv * rot(u_time * (1.0 + u_distortion) * 0.2 + i)) - 0.5;
                        float d = length(gv);
                        col += (0.5 + 0.5 * cos(u_time + vec3(0,2,4))) * smoothstep(0.4, 0.1, d);
                    }
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            ShaderType.PLASMA: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_iterations;
                uniform float u_brightness;
                out vec4 f_color;

                void main() {
                    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * 2.0;
                    uv.x *= u_resolution.x / u_resolution.y;
                    uv *= u_zoom * 5.0;
                    
                    float v = 0.0;
                    for (float i = 1.0; i <= u_iterations; i++) {
                        v += sin(uv.x * i + u_time) + sin(uv.y * i + u_time);
                        uv = mat2(cos(i), -sin(i), sin(i), cos(i)) * uv;
                    }
                    
                    vec3 col = 0.5 + 0.5 * cos(v * 3.14 + u_time + vec3(0, 2, 4));
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            ShaderType.CLOUD: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                out vec4 f_color;

                mat2 rot(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom * 2.0;
                    float t = u_time * 0.1;
                    vec3 col = vec3(0.0);
                    
                    for (float i = 1.0; i <= u_iterations; i++) {
                        uv += sin(uv.yx * (2.0 + u_distortion) + t + i) * 0.4;
                        uv *= rot(t * 0.05 + i);
                        float d = length(uv);
                        float val = smoothstep(0.0, 0.8, abs(sin(d * (10.0 + u_distortion) - t * 2.0)));
                        col += (0.5 + 0.5 * cos(t + d + vec3(0, 2, 4))) * val / i;
                    }
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            ShaderType.MANDALA: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                out vec4 f_color;

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y;
                    float breath = sin(u_time * 0.2) * 0.2 + 1.0;
                    uv *= u_zoom * 2.0 * breath;
                    
                    float angle = atan(uv.y, uv.x);
                    float dist = length(uv);
                    vec3 col = vec3(0.0);
                    
                    for (float i = 0.0; i < u_iterations; i++) {
                        float t = u_time * 0.1 * (i + 1.0) * (1.0 + u_distortion * 0.5);
                        float segments = 6.0 + i * 2.0;
                        float a = mod(angle + t, 6.28 / segments) - 3.14 / segments;
                        vec2 p = vec2(cos(a), sin(a)) * dist;
                        float d = sin(length(p - vec2(0.5, 0.0)) * 20.0 - u_time * 2.0);
                        col += (0.5 + 0.5 * cos(u_time * 0.5 + i + dist + vec3(0, 2, 4))) * smoothstep(0.1, 0.2, abs(d));
                    }
                    col *= 1.0 - smoothstep(0.5, 1.5, dist);
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            ShaderType.GALAXY: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                out vec4 f_color;

                float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5); }
                mat2 rot(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom;
                    float t = u_time * 0.5;
                    uv *= rot(t * 0.1);
                    
                    float angle = atan(uv.y, uv.x);
                    float dist = length(uv);
                    vec3 col = vec3(0.0);
                    
                    for (float i = 0.0; i < u_iterations; i++) {
                        float armAngle = angle + dist * (2.0 + u_distortion) - t * (0.3 + i * 0.05);
                        float armDist = sin(armAngle * (3.0 + i)) * 0.3;
                        float arm = smoothstep(0.2, 0.0, abs(dist - 0.5 - armDist));
                        
                        vec2 starCoord = vec2(angle * 5.0 + i, dist * 10.0);
                        float stars = smoothstep(0.98, 1.0, hash(floor(starCoord + t * 0.1))) * smoothstep(0.3, 0.8, dist);
                        
                        col += (0.5 + 0.5 * cos(t * 0.3 + i * 2.0 + vec3(0, 2, 4))) * arm + vec3(1.0, 0.95, 0.9) * stars;
                    }
                    col += vec3(1.0, 0.8, 0.6) * exp(-dist * 3.0) * 0.5;
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            ShaderType.TECTONIC: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                out vec4 f_color;

                float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5); }
                float fbm(vec2 p) {
                    float v = 0.0, a = 0.5;
                    for (int i = 0; i < 6; i++) { v += a * hash(p); p *= 2.0; a *= 0.5; }
                    return v;
                }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom * 2.0;
                    float t = u_time * 0.1;
                    vec3 col = vec3(0.0);
                    
                    for (float i = 0.0; i < u_iterations; i++) {
                        vec2 drift = vec2(sin(t * 0.2 + i), cos(t * 0.15 + i * 1.5)) * t * 0.05;
                        vec2 pUV = uv + drift;
                        pUV += vec2(fbm(pUV * (2.0 + u_distortion) + t * 0.1), fbm(pUV * (2.0 + u_distortion) + t * 0.1 + 100.0)) * 0.3;
                        float e = fbm(pUV);
                        
                        vec3 c = e < 0.4 ? vec3(0.1, 0.2, 0.5) : e < 0.6 ? vec3(0.2, 0.6, 0.3) : vec3(0.7, 0.7, 0.8);
                        c *= 0.5 + 0.5 * cos(t * 0.1 + e);
                        col += c / (i + 1.0);
                    }
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            ShaderType.BIOLUMINESCENT: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                out vec4 f_color;

                float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5); }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom * 3.0;
                    float t = u_time * 0.5;
                    uv.y += sin(t * 0.1) * 0.3;
                    vec3 col = vec3(0.0, 0.05, 0.15);
                    
                    for (float i = 0.0; i < u_iterations; i++) {
                        vec2 c = uv * (5.0 + i * 2.0);
                        c.x += t * (0.1 + i * 0.05);
                        c.y += sin(c.x * 2.0 + t * 0.2) * (0.2 + u_distortion * 0.2);
                        
                        vec2 id = floor(c);
                        vec2 gv = fract(c) - 0.5;
                        float h = hash(id + i);
                        float pulse = smoothstep(0.3, 0.8, sin(t * (0.5 + h * 2.0) + h * 6.28) * 0.5 + 0.5);
                        float org = smoothstep(0.3, 0.1, length(gv)) * pulse;
                        
                        col += (0.5 + 0.5 * cos(h * 6.28 + vec3(0, 2, 4))) * org / sqrt(i + 1.0);
                    }
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            ShaderType.AURORA: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                out vec4 f_color;

                float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5); }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom;
                    float t = u_time * 0.6;
                    float season = sin(t * 0.05) * 0.5 + 0.5;
                    vec3 col = vec3(0.01, 0.01, 0.03);
                    
                    for (float i = 0.0; i < u_iterations; i++) {
                        float x = uv.x * (3.0 + i);
                        float wave = sin(x + t * (0.3 + i * 0.1)) * (0.5 + u_distortion);
                        float curtain = abs(uv.y - wave);
                        float intensity = smoothstep(0.5, 0.0, curtain) * (0.5 + season * 0.5);
                        float shimmer = hash(vec2(x * 20.0, t * 2.0 + i));
                        intensity *= 0.7 + shimmer * 0.3;
                        
                        col += (0.5 + 0.5 * cos(t * 2.0 + i * 1.5 + vec3(0, 2, 4))) * intensity;
                    }
                    col += smoothstep(0.99, 1.0, hash(uv * 100.0)) * 0.3;
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            ShaderType.CRYSTAL: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                out vec4 f_color;

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y;
                    float t = u_time * 0.2;
                    float growth = smoothstep(0.0, 10.0, t) * (1.0 + sin(t * 0.1) * 0.2);
                    uv *= u_zoom * (1.0 + growth * 0.5);

                    float angle = atan(uv.y, uv.x);
                    float dist = length(uv);
                    vec3 col = vec3(0.0);

                    for (float i = 0.0; i < u_iterations; i++) {
                        float a = mod(angle + t * 0.05 * (i + 1.0), 6.28 / 6.0);
                        float face = abs(sin(a * 3.0 + t * 0.1));
                        float layer = sin(dist * (10.0 + i * 3.0) - t * 0.3 + face * u_distortion) * 0.5 + 0.5;
                        layer *= smoothstep(i * 0.5, i * 0.5 + 2.0, t);
                        col += (0.5 + 0.5 * cos(dist + i + vec3(0, 2, 4))) * layer * smoothstep(1.0, 0.5, dist);
                    }
                    col += vec3(1.0, 1.0, 0.9) * exp(-dist * 5.0);
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            ShaderType.MANDELBROT: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_color_shift;
                uniform float u_brightness;
                uniform float u_hue_shift;
                uniform float u_saturation;
                uniform float u_scroll_x;
                uniform float u_scroll_y;
                uniform float u_rotation;
                out vec4 f_color;

                mat2 rotate2d(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

                vec3 palette(float t) {
                    return 0.5 + 0.5 * cos(6.28318 * (t * u_color_shift + vec3(0.0, 0.33, 0.67) + u_hue_shift));
                }

                void main() {
                    vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution) / u_resolution.y;
                    uv *= rotate2d(u_rotation);
                    uv = uv * 2.5 / u_zoom + vec2(u_scroll_x - 0.5, u_scroll_y);

                    // Power controlled by distortion: 2.0 (classic) to 4.0 (Multibrot)
                    float power = 2.0 + u_distortion * 2.0;
                    float maxIter = u_iterations * 20.0;
                    vec2 z = vec2(0.0);
                    float iter = 0.0;

                    for (float i = 0.0; i < 200.0; i++) {
                        if (i >= maxIter || dot(z, z) > 4.0) break;
                        float r = length(z);
                        float theta = atan(z.y, z.x);
                        z = pow(r, power) * vec2(cos(power * theta), sin(power * theta)) + uv;
                        iter++;
                    }

                    if (iter >= maxIter) {
                        f_color = vec4(0.0, 0.0, 0.0, 1.0);
                        return;
                    }

                    float smooth_iter = (iter - log2(max(1.0, log2(dot(z, z))))) / maxIter;
                    vec3 col = palette(smooth_iter + u_time * 0.05);
                    float gray = dot(col, vec3(0.299, 0.587, 0.114));
                    col = mix(vec3(gray), col, u_saturation);
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            ShaderType.JULIA: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_color_shift;
                uniform float u_brightness;
                uniform float u_hue_shift;
                uniform float u_saturation;
                uniform float u_scroll_x;
                uniform float u_scroll_y;
                uniform float u_rotation;
                out vec4 f_color;

                mat2 rotate2d(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

                vec3 palette(float t) {
                    return 0.5 + 0.5 * cos(6.28318 * (t * u_color_shift + vec3(0.0, 0.33, 0.67) + u_hue_shift));
                }

                void main() {
                    vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution) / u_resolution.y;
                    uv *= rotate2d(u_rotation);
                    uv *= 1.5 / u_zoom;
                    uv += vec2(u_scroll_x * 0.1, u_scroll_y * 0.1);

                    // c orbits a circle of radius 0.7885 (near chaotic boundary).
                    // scroll_x offsets the phase; distortion controls animation speed.
                    float angle = u_time * u_distortion * 0.4 + u_scroll_x * 3.14159;
                    vec2 c = 0.7885 * vec2(cos(angle), sin(angle));

                    float maxIter = u_iterations * 20.0;
                    vec2 z = uv;
                    float iter = 0.0;

                    for (float i = 0.0; i < 200.0; i++) {
                        if (i >= maxIter || dot(z, z) > 4.0) break;
                        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
                        iter++;
                    }

                    if (iter >= maxIter) {
                        f_color = vec4(0.0, 0.0, 0.0, 1.0);
                        return;
                    }

                    float smooth_iter = (iter - log2(max(1.0, log2(dot(z, z))))) / maxIter;
                    vec3 col = palette(smooth_iter + u_time * 0.05);
                    float gray = dot(col, vec3(0.299, 0.587, 0.114));
                    col = mix(vec3(gray), col, u_saturation);
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            ShaderType.BURNING_SHIP: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_color_shift;
                uniform float u_brightness;
                uniform float u_hue_shift;
                uniform float u_saturation;
                uniform float u_scroll_x;
                uniform float u_scroll_y;
                uniform float u_rotation;
                out vec4 f_color;

                mat2 rotate2d(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

                vec3 palette(float t) {
                    return 0.5 + 0.5 * cos(6.28318 * (t * u_color_shift + vec3(0.0, 0.33, 0.67) + u_hue_shift));
                }

                void main() {
                    vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution) / u_resolution.y;
                    uv *= rotate2d(u_rotation);
                    uv = uv * 2.5 / u_zoom + vec2(u_scroll_x - 0.4, u_scroll_y - 0.3);

                    // distortion morphs between Mandelbrot (0) and full Burning Ship (1)
                    float maxIter = u_iterations * 20.0;
                    vec2 z = vec2(0.0);
                    float iter = 0.0;

                    for (float i = 0.0; i < 200.0; i++) {
                        if (i >= maxIter || dot(z, z) > 4.0) break;
                        float zx = mix(z.x, abs(z.x), u_distortion);
                        float zy = mix(z.y, abs(z.y), u_distortion);
                        z = vec2(zx*zx - zy*zy, 2.0*zx*zy) + uv;
                        iter++;
                    }

                    if (iter >= maxIter) {
                        f_color = vec4(0.0, 0.0, 0.0, 1.0);
                        return;
                    }

                    float smooth_iter = (iter - log2(max(1.0, log2(dot(z, z))))) / maxIter;
                    vec3 col = palette(smooth_iter + u_time * 0.05);
                    float gray = dot(col, vec3(0.299, 0.587, 0.114));
                    col = mix(vec3(gray), col, u_saturation);
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            ShaderType.NEWTON: '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_color_shift;
                uniform float u_brightness;
                uniform float u_hue_shift;
                uniform float u_saturation;
                uniform float u_scroll_x;
                uniform float u_scroll_y;
                uniform float u_rotation;
                out vec4 f_color;

                mat2 rotate2d(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

                vec2 cmul(vec2 a, vec2 b) { return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
                vec2 cdiv(vec2 a, vec2 b) { float d = dot(b, b); return vec2(dot(a, b), a.y*b.x - a.x*b.y) / d; }

                void main() {
                    vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution) / u_resolution.y;
                    uv *= rotate2d(u_rotation);
                    uv = uv * 2.0 / u_zoom + vec2(u_scroll_x, u_scroll_y);

                    // Newton's method for z^3 - 1 = 0
                    // Roots: (1,0), (-0.5, 0.866), (-0.5, -0.866)
                    vec2 r0 = vec2(1.0, 0.0);
                    vec2 r1 = vec2(-0.5,  0.8660254);
                    vec2 r2 = vec2(-0.5, -0.8660254);

                    float maxIter = u_iterations * 20.0;
                    vec2 z = uv;
                    float iter = 0.0;
                    int root = -1;

                    for (float i = 0.0; i < 200.0; i++) {
                        if (i >= maxIter) break;
                        vec2 z2 = cmul(z, z);
                        vec2 z3 = cmul(z2, z);
                        // z = (2*z^3 + 1) / (3*z^2)  + distortion perturbation
                        vec2 num = 2.0 * z3 + vec2(1.0 + u_distortion * sin(u_time), u_distortion * cos(u_time * 0.7));
                        vec2 den = 3.0 * z2;
                        if (dot(den, den) < 1e-10) break;
                        z = cdiv(num, den);
                        iter++;

                        if (length(z - r0) < 0.001) { root = 0; break; }
                        if (length(z - r1) < 0.001) { root = 1; break; }
                        if (length(z - r2) < 0.001) { root = 2; break; }
                    }

                    float shade = 1.0 - iter / maxIter;
                    float hue = float(root) / 3.0 + u_hue_shift + u_time * 0.05;
                    vec3 col = (0.5 + 0.5 * cos(6.28318 * (hue * u_color_shift + vec3(0.0, 0.33, 0.67)))) * shade;
                    float gray = dot(col, vec3(0.299, 0.587, 0.114));
                    col = mix(vec3(gray), col, u_saturation);
                    f_color = vec4(col * u_brightness, 1.0);
                }
            '''
        }
        return vertex_shader, code
