import moderngl
import numpy as np
import cv2
import time
from param import ParamTable

class ShaderVisualizer:
    def __init__(self, params, width=1280, height=720):
        self.width = width
        self.height = height
        self.ctx = moderngl.create_context(standalone=True)
        subclass = self.__class__.__name__

        self.time = time.time()

        self.zoom = params.add("s_zoom", 0.1, 5.0, 1.5, subclass)
        self.distortion = params.add("s_distortion", 0.0, 1.0, 0.5, subclass)
        self.iterations = params.add("s_iterations", 1.0, 10.0, 4.0, subclass)
        self.color_shift = params.add("s_color_shift", 0.5, 3.0, 1.0, subclass)
        self.brightness = params.add("s_brightness", 0.0, 2.0, 1.0, subclass)
        self.hue_shift = params.add("s_hue_shift", 0.0, 7, 0.0, subclass) # 6.28
        self.saturation = params.add("s_saturation", 0.0, 2.0, 1.0, subclass)
        self.x_shift = params.add("s_x_shift", -5.0, 5.0, 0.0, subclass)
        self.y_shift = params.add("s_y_shift", -5.0, 5.0, 0.0, subclass)
        self.rotation = params.add("s_rotation", -3.14, 3.14, 0.0, subclass)
        self.speed = params.add("s_speed", 0.0, 2.0, 1.0, subclass)

        # Vertex shader
        vertex_shader = '''
            in vec2 in_vert;
            void main() {
                gl_Position = vec4(in_vert, 0.0, 1.0);
            }
        '''

        # All fragment shaders with proper u_time usage
        shaders_code = {
            'Fractal': '''
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
                uniform float u_x_shift;
                uniform float u_y_shift;
                uniform float u_rotation;
                uniform float u_speed;
                out vec4 f_color;

                mat2 rot(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

                vec3 rgb2hsv(vec3 c) {
                    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
                    float d = q.x - min(q.w, q.y);
                    float e = 1.0e-10;
                    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
                }

                vec3 hsv2rgb(vec3 c) {
                    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                }

                vec3 palette(float t) {
                    return 0.5 + 0.5 * cos(6.28 * (t * u_color_shift + vec3(0.0, 0.33, 0.67)));
                }

                void main() {
                    vec2 uv = (gl_FragCoord.xy * 2.0 - u_resolution) / u_resolution.y;
                    uv *= rot(u_rotation);
                    uv += vec2(u_x_shift, u_y_shift);
                    vec2 uv0 = uv;
                    vec3 col = vec3(0.0);
                    float t = u_time * u_speed;
                    
                    for (float i = 0.0; i < u_iterations; i++) {
                        uv = fract(uv * u_zoom) - 0.5;
                        float d = length(uv) * exp(-length(uv0));
                        col += palette(length(uv0) + i * 0.4 + t * 0.4) * (0.01 / abs(d));
                    }

                    vec3 hsv = rgb2hsv(col);
                    hsv.x += u_hue_shift / 7.0;
                    hsv.y *= u_saturation;
                    col = hsv2rgb(hsv);

                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            'Grid': '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                uniform float u_hue_shift;
                uniform float u_saturation;
                uniform float u_x_shift;
                uniform float u_y_shift;
                uniform float u_rotation;
                uniform float u_speed;
                out vec4 f_color;

                mat2 rot(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }
                
                vec3 rgb2hsv(vec3 c) {
                    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
                    float d = q.x - min(q.w, q.y);
                    float e = 1.0e-10;
                    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
                }

                vec3 hsv2rgb(vec3 c) {
                    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom * 3.0;
                    uv *= rot(u_rotation);
                    uv += vec2(u_x_shift, u_y_shift);
                    vec3 col = vec3(0.0);
                    float t = u_time * u_speed;
                    
                    for(float i = 0.0; i < u_iterations; i++){
                        vec2 gv = fract(uv * rot(t * (1.0 + u_distortion) * 0.2 + i)) - 0.5;
                        float d = length(gv);
                        col += (0.5 + 0.5 * cos(t + vec3(0,2,4))) * smoothstep(0.4, 0.1, d);
                    }

                    vec3 hsv = rgb2hsv(col);
                    hsv.x += u_hue_shift / 7.0;
                    hsv.y *= u_saturation;
                    col = hsv2rgb(hsv);

                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            'Plasma': '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_iterations;
                uniform float u_brightness;
                uniform float u_hue_shift;
                uniform float u_saturation;
                uniform float u_x_shift;
                uniform float u_y_shift;
                uniform float u_rotation;
                uniform float u_speed;
                out vec4 f_color;

                mat2 rot(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

                vec3 rgb2hsv(vec3 c) {
                    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
                    float d = q.x - min(q.w, q.y);
                    float e = 1.0e-10;
                    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
                }

                vec3 hsv2rgb(vec3 c) {
                    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                }

                void main() {
                    vec2 uv = (gl_FragCoord.xy / u_resolution - 0.5) * 2.0;
                    uv.x *= u_resolution.x / u_resolution.y;
                    uv *= rot(u_rotation);
                    uv += vec2(u_x_shift, u_y_shift);
                    uv *= u_zoom * 5.0;
                    float t = u_time * u_speed;
                    
                    float v = 0.0;
                    for (float i = 1.0; i <= u_iterations; i++) {
                        v += sin(uv.x * i + t) + sin(uv.y * i + t);
                        uv = rot(i) * uv;
                    }
                    
                    vec3 col = 0.5 + 0.5 * cos(v * 3.14 + t + vec3(0, 2, 4));
                    
                    vec3 hsv = rgb2hsv(col);
                    hsv.x += u_hue_shift / 7.0;
                    hsv.y *= u_saturation;
                    col = hsv2rgb(hsv);
                    
                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            'Cloud': '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                uniform float u_hue_shift;
                uniform float u_saturation;
                uniform float u_x_shift;
                uniform float u_y_shift;
                uniform float u_rotation;
                uniform float u_speed;
                out vec4 f_color;

                mat2 rot(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

                vec3 rgb2hsv(vec3 c) {
                    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
                    float d = q.x - min(q.w, q.y);
                    float e = 1.0e-10;
                    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
                }

                vec3 hsv2rgb(vec3 c) {
                    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom * 2.0;
                    uv *= rot(u_rotation);
                    uv += vec2(u_x_shift, u_y_shift);
                    float t = u_time * u_speed * 0.1;
                    vec3 col = vec3(0.0);
                    
                    for(float i = 1.0; i <= u_iterations; i++){
                        uv += sin(uv.yx * (2.0 + u_distortion) + t + i) * 0.4;
                        uv *= rot(t * 0.05 + i);
                        float d = length(uv);
                        float val = smoothstep(0.0, 0.8, abs(sin(d * (10.0 + u_distortion) - t * 2.0)));
                        col += (0.5 + 0.5 * cos(t + d + vec3(0, 2, 4))) * val / i;
                    }

                    vec3 hsv = rgb2hsv(col);
                    hsv.x += u_hue_shift / 7.0;
                    hsv.y *= u_saturation;
                    col = hsv2rgb(hsv);

                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            'Mandala': '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                uniform float u_hue_shift;
                uniform float u_saturation;
                uniform float u_x_shift;
                uniform float u_y_shift;
                uniform float u_rotation;
                uniform float u_speed;
                out vec4 f_color;

                mat2 rot(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

                vec3 rgb2hsv(vec3 c) {
                    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
                    float d = q.x - min(q.w, q.y);
                    float e = 1.0e-10;
                    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
                }

                vec3 hsv2rgb(vec3 c) {
                    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y;
                    uv *= rot(u_rotation);
                    uv += vec2(u_x_shift, u_y_shift);
                    float t = u_time * u_speed;
                    float breath = sin(t * 0.2) * 0.2 + 1.0;
                    uv *= u_zoom * 2.0 * breath;
                    
                    float angle = atan(uv.y, uv.x);
                    float dist = length(uv);
                    vec3 col = vec3(0.0);
                    
                    for(float i = 0.0; i < u_iterations; i++){
                        float time_iter = t * 0.1 * (i + 1.0) * (1.0 + u_distortion * 0.5);
                        float segments = 6.0 + i * 2.0;
                        float a = mod(angle + time_iter, 6.28 / segments) - 3.14 / segments;
                        vec2 p = vec2(cos(a), sin(a)) * dist;
                        float d = sin(length(p - vec2(0.5, 0.0)) * 20.0 - t * 2.0);
                        col += (0.5 + 0.5 * cos(t * 0.5 + i + dist + vec3(0,2,4))) * smoothstep(0.1, 0.2, abs(d));
                    }
                    col *= 1.0 - smoothstep(0.5, 1.5, dist);

                    vec3 hsv = rgb2hsv(col);
                    hsv.x += u_hue_shift / 7.0;
                    hsv.y *= u_saturation;
                    col = hsv2rgb(hsv);

                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            'Galaxy': '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                uniform float u_hue_shift;
                uniform float u_saturation;
                uniform float u_x_shift;
                uniform float u_y_shift;
                uniform float u_rotation;
                uniform float u_speed;
                out vec4 f_color;

                float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5); }
                mat2 rot(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

                vec3 rgb2hsv(vec3 c) {
                    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
                    float d = q.x - min(q.w, q.y);
                    float e = 1.0e-10;
                    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
                }

                vec3 hsv2rgb(vec3 c) {
                    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom;
                    uv *= rot(u_rotation);
                    uv += vec2(u_x_shift, u_y_shift);
                    float t = u_time * u_speed * 0.5;
                    uv *= rot(t * 0.1);
                    
                    float angle = atan(uv.y, uv.x);
                    float dist = length(uv);
                    vec3 col = vec3(0.0);
                    
                    for(float i = 0.0; i < u_iterations; i++){
                        float armAngle = angle + dist * (2.0 + u_distortion) - t * (0.3 + i * 0.05);
                        float armDist = sin(armAngle * (3.0 + i)) * 0.3;
                        float arm = smoothstep(0.2, 0.0, abs(dist - 0.5 - armDist));
                        
                        vec2 starCoord = vec2(angle * 5.0 + i, dist * 10.0);
                        float stars = smoothstep(0.98, 1.0, hash(floor(starCoord + t * 0.1))) * smoothstep(0.3, 0.8, dist);
                        
                        col += (0.5 + 0.5 * cos(t * 0.3 + i * 2.0 + vec3(0,2,4))) * arm + vec3(1.0, 0.95, 0.9) * stars;
                    }
                    col += vec3(1.0, 0.8, 0.6) * exp(-dist * 3.0) * 0.5;

                    vec3 hsv = rgb2hsv(col);
                    hsv.x += u_hue_shift / 7.0;
                    hsv.y *= u_saturation;
                    col = hsv2rgb(hsv);

                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            'Tectonic': '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                uniform float u_hue_shift;
                uniform float u_saturation;
                uniform float u_x_shift;
                uniform float u_y_shift;
                uniform float u_rotation;
                uniform float u_speed;
                out vec4 f_color;

                float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5); }
                float fbm(vec2 p) {
                    float v = 0.0, a = 0.5;
                    for(int i = 0; i < 6; i++) { v += a * hash(p); p *= 2.0; a *= 0.5; }
                    return v;
                }
                mat2 rot(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

                vec3 rgb2hsv(vec3 c) {
                    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
                    float d = q.x - min(q.w, q.y);
                    float e = 1.0e-10;
                    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
                }

                vec3 hsv2rgb(vec3 c) {
                    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom * 2.0;
                    uv *= rot(u_rotation);
                    uv += vec2(u_x_shift, u_y_shift);
                    float t = u_time * u_speed * 0.1;
                    vec3 col = vec3(0.0);
                    
                    for(float i = 0.0; i < u_iterations; i++){
                        vec2 drift = vec2(sin(t * 0.2 + i), cos(t * 0.15 + i * 1.5)) * t * 0.05;
                        vec2 pUV = uv + drift;
                        pUV += vec2(fbm(pUV * (2.0 + u_distortion) + t * 0.1), fbm(pUV * (2.0 + u_distortion) + t * 0.1 + 100.0)) * 0.3;
                        float e = fbm(pUV);
                        
                        vec3 c = e < 0.4 ? vec3(0.1, 0.2, 0.5) : e < 0.6 ? vec3(0.2, 0.6, 0.3) : vec3(0.7, 0.7, 0.8);
                        c *= 0.5 + 0.5 * cos(t * 0.1 + e);
                        col += c / (i + 1.0);
                    }

                    vec3 hsv = rgb2hsv(col);
                    hsv.x += u_hue_shift / 7.0;
                    hsv.y *= u_saturation;
                    col = hsv2rgb(hsv);

                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            'Bioluminescent': '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                uniform float u_hue_shift;
                uniform float u_saturation;
                uniform float u_x_shift;
                uniform float u_y_shift;
                uniform float u_rotation;
                uniform float u_speed;
                out vec4 f_color;

                float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5); }
                mat2 rot(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

                vec3 rgb2hsv(vec3 c) {
                    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
                    float d = q.x - min(q.w, q.y);
                    float e = 1.0e-10;
                    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
                }

                vec3 hsv2rgb(vec3 c) {
                    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom * 3.0;
                    uv *= rot(u_rotation);
                    uv += vec2(u_x_shift, u_y_shift);
                    float t = u_time * u_speed * 0.5;
                    uv.y += sin(t * 0.1) * 0.3;
                    vec3 col = vec3(0.0, 0.05, 0.15);
                    
                    for(float i = 0.0; i < u_iterations; i++){
                        vec2 c = uv * (5.0 + i * 2.0);
                        c.x += t * (0.1 + i * 0.05);
                        c.y += sin(c.x * 2.0 + t * 0.2) * (0.2 + u_distortion * 0.2);
                        
                        vec2 id = floor(c);
                        vec2 gv = fract(c) - 0.5;
                        float h = hash(id + i);
                        float pulse = smoothstep(0.3, 0.8, sin(t * (0.5 + h * 2.0) + h * 6.28) * 0.5 + 0.5);
                        float org = smoothstep(0.3, 0.1, length(gv)) * pulse;
                        
                        col += (0.5 + 0.5 * cos(h * 6.28 + vec3(0,2,4))) * org / sqrt(i + 1.0);
                    }

                    vec3 hsv = rgb2hsv(col);
                    hsv.x += u_hue_shift / 7.0;
                    hsv.y *= u_saturation;
                    col = hsv2rgb(hsv);

                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            'Aurora': '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                uniform float u_hue_shift;
                uniform float u_saturation;
                uniform float u_x_shift;
                uniform float u_y_shift;
                uniform float u_rotation;
                uniform float u_speed;
                out vec4 f_color;

                float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5); }
                mat2 rot(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

                vec3 rgb2hsv(vec3 c) {
                    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
                    float d = q.x - min(q.w, q.y);
                    float e = 1.0e-10;
                    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
                }

                vec3 hsv2rgb(vec3 c) {
                    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y * u_zoom;
                    uv *= rot(u_rotation);
                    uv += vec2(u_x_shift, u_y_shift);
                    float t = u_time * u_speed * 0.6;
                    float season = sin(t * 0.05) * 0.5 + 0.5;
                    vec3 col = vec3(0.01, 0.01, 0.03);
                    
                    for(float i = 0.0; i < u_iterations; i++){
                        float x = uv.x * (3.0 + i);
                        float wave = sin(x + t * (0.3 + i * 0.1)) * (0.5 + u_distortion);
                        float curtain = abs(uv.y - wave);
                        float intensity = smoothstep(0.5, 0.0, curtain) * (0.5 + season * 0.5);
                        float shimmer = hash(vec2(x * 20.0, t * 2.0 + i));
                        intensity *= 0.7 + shimmer * 0.3;
                        
                        col += (0.5 + 0.5 * cos(t * 0.2 + i * 1.5 + vec3(0,2,4))) * intensity;
                    }
                    col += smoothstep(0.99, 1.0, hash(uv * 100.0)) * 0.3;
                    
                    vec3 hsv = rgb2hsv(col);
                    hsv.x += u_hue_shift / 7.0;
                    hsv.y *= u_saturation;
                    col = hsv2rgb(hsv);

                    f_color = vec4(col * u_brightness, 1.0);
                }
            ''',
            'Crystal': '''
                #version 330
                uniform vec2 u_resolution;
                uniform float u_time;
                uniform float u_zoom;
                uniform float u_distortion;
                uniform float u_iterations;
                uniform float u_brightness;
                uniform float u_hue_shift;
                uniform float u_saturation;
                uniform float u_x_shift;
                uniform float u_y_shift;
                uniform float u_rotation;
                uniform float u_speed;
                out vec4 f_color;

                mat2 rot(float a) { return mat2(cos(a), -sin(a), sin(a), cos(a)); }

                vec3 rgb2hsv(vec3 c) {
                    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
                    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
                    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
                    float d = q.x - min(q.w, q.y);
                    float e = 1.0e-10;
                    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
                }

                vec3 hsv2rgb(vec3 c) {
                    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                }

                void main() {
                    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y;
                    uv *= rot(u_rotation);
                    uv += vec2(u_x_shift, u_y_shift);
                    float t = u_time * u_speed * 0.2;
                    float growth = smoothstep(0.0, 10.0, t) * (1.0 + sin(t * 0.1) * 0.2);
                    uv *= u_zoom * (1.0 + growth * 0.5);
                    
                    float angle = atan(uv.y, uv.x);
                    float dist = length(uv);
                    vec3 col = vec3(0.0);
                    
                    for(float i = 0.0; i < u_iterations; i++){
                        float a = mod(angle + t * 0.05 * (i + 1.0), 6.28 / 6.0);
                        float face = abs(sin(a * 3.0 + t * 0.1));
                        float layer = sin(dist * (10.0 + i * 3.0) - t * 0.3 + face * u_distortion) * 0.5 + 0.5;
                        layer *= smoothstep(i * 0.5, i * 0.5 + 2.0, t);
                        col += (0.5 + 0.5 * cos(dist + i + vec3(0,2,4))) * layer * smoothstep(1.0, 0.5, dist);
                    }
                    col += vec3(1.0, 1.0, 0.9) * exp(-dist * 5.0);

                    vec3 hsv = rgb2hsv(col);
                    hsv.x += u_hue_shift / 7.0;
                    hsv.y *= u_saturation;
                    col = hsv2rgb(hsv);

                    f_color = vec4(col * u_brightness, 1.0);
                }
            '''
        }

        # Compile programs
        self.programs = {}
        vertices = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4')
        vbo = self.ctx.buffer(vertices)
        
        for name, frag_code in shaders_code.items():
            prog = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=frag_code)
            vao = self.ctx.vertex_array(prog, [(vbo, '2f', 'in_vert')])
            self.programs[name] = {'prog': prog, 'vao': vao}
        
        self.shader_names = list(shaders_code.keys())
        self.current_shader = self.shader_names[0]
        
        # FBO
        self.fbo_texture = self.ctx.texture((width, height), 3)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.fbo_texture])

    def set_shader(self, index):
        if 0 <= index < len(self.shader_names):
            self.current_shader = self.shader_names[index]
            print(f"Switched to: {self.current_shader}")

    def render(self, params):
        prog = self.programs[self.current_shader]['prog']
        vao = self.programs[self.current_shader]['vao']
        
        # Set all uniforms
        for name, value in params.items():
            if name in prog:
                prog[name].value = value
        
        self.ctx.clear(0, 0, 0)
        self.fbo.use()
        vao.render(moderngl.TRIANGLE_STRIP)
        
        data = np.frombuffer(self.fbo.read(components=3), dtype=np.uint8)
        img = data.reshape((self.height, self.width, 3))
        return cv2.flip(img, 0)



if __name__ == "__main__":
    WIDTH, HEIGHT = 1280, 720
    viz = ShaderVisualizer(WIDTH, HEIGHT)
    start_time = viz.time
    
    params = {
        'u_resolution': (WIDTH, HEIGHT),
        'u_time': 0.0,
        'u_zoom': viz.zoom.value,
        'u_distortion': viz.distortion.value,
        'u_iterations': viz.iterations.value,
        'u_color_shift': viz.color_shift.value,
        'u_brightness': viz.brightness.value,
        'u_hue_shift': viz.hue_shift.value,
        'u_saturation': viz.saturation.value,
        'u_x_shift': viz.x_shift.value,
        'u_y_shift': viz.y_shift.value,
        'u_rotation': viz.rotation.value,
        'u_speed': viz.speed.value
    }
    
    while True:
        params['u_time'] = time.time() - start_time
        frame = viz.render(params)
        
        cv2.putText(frame, f"{viz.current_shader}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Shader Visualizer", frame)
        
        key = cv2.waitKey(16) & 0xFF
        if key == 27:
            break
        elif key == ord('1'):
            viz.set_shader(0)
        elif key == ord('2'):
            viz.set_shader(1)
        elif key == ord('3'):
            viz.set_shader(2)
        elif key == ord('4'):
            viz.set_shader(3)
        elif key == ord('5'):
            viz.set_shader(4)
        elif key == ord('6'):
            viz.set_shader(5)
        elif key == ord('7'):
            viz.set_shader(6)
        elif key == ord('8'):
            viz.set_shader(7)
        elif key == ord('9'):
            viz.set_shader(8)
        elif key == ord('0'):
            viz.set_shader(9)
        elif key == ord('r'):
            params.update({'u_zoom': 1.5, 'u_distortion': 0.5, 'u_iterations': 4.0, 
                          'u_color_shift': 1.0, 'u_brightness': 1.0})
    
    cv2.destroyAllWindows()