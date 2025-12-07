import moderngl
import numpy as np
import cv2
import time

class ShaderVisualizer:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        
        # Create a standalone context (headless)
        try:
            self.ctx = moderngl.create_context(standalone=True)
        except Exception as e:
            print(f"Error creating standalone context: {e}")
            print("Ensure you have valid OpenGL drivers installed.")
            raise

        # ------------------------------------------------------------------
        # VERTEX SHADER
        # ------------------------------------------------------------------
        self.vertex_shader = '''
            #version 330
            in vec2 in_vert;
            out vec2 v_uv;
            void main() {
                v_uv = in_vert * 0.5 + 0.5;
                gl_Position = vec4(in_vert, 0.0, 1.0);
            }
        '''

        # ------------------------------------------------------------------
        # FRAGMENT SHADERS
        # ------------------------------------------------------------------

        # --- Shader 1: Raymarching/SDF Fractal ---
        shader_fractal = '''
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
            }
        '''

        # --- Shader 2: Hypnotic Grid / Voronoi-ish ---
        shader_grid = '''
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

            mat2 rotate2d(float angle){
                return mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
            }

            void main() {
                vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;
                
                // Apply rotation
                uv *= rotate2d(u_rotation);
                
                // Apply scrolling
                uv += vec2(u_scroll_x, u_scroll_y);
                
                vec3 finalColor = vec3(0.0);
                
                uv *= (u_zoom * 3.0);
                
                vec2 gv = fract(uv) - 0.5;
                vec2 id = floor(uv);
                
                for(float i=0.0; i<u_iterations; i++){
                    float t = u_time * (1.0 + u_distortion);
                    gv *= rotate2d(t * 0.2 + i);
                    
                    float d = length(gv - vec2(sin(t)*0.1, cos(t)*0.1));
                    float mask = smoothstep(0.4 + u_distortion*0.1, 0.1, d);
                    
                    vec3 col = 0.5 + 0.5 * cos(u_time + uv.xyx + vec3(0,2,4) * u_color_shift + u_hue_shift);
                    finalColor += col * mask;
                }
                
                // Apply saturation
                float gray = dot(finalColor, vec3(0.299, 0.587, 0.114));
                finalColor = mix(vec3(gray), finalColor, u_saturation);
                
                f_color = vec4(finalColor * u_brightness, 1.0);
            }
        '''

        # --- Shader 3: Liquid Plasma ---
        shader_plasma = '''
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

            mat2 rotate2d(float angle){
                return mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
            }

            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution.xy;
                uv = uv * 2.0 - 1.0;
                uv.x *= u_resolution.x / u_resolution.y;
                
                // Apply rotation
                uv *= rotate2d(u_rotation);
                
                // Apply scrolling
                uv += vec2(u_scroll_x, u_scroll_y);
                
                vec2 coord = uv * (u_zoom * 5.0);
                
                float v = 0.0;
                
                for (float i = 1.0; i <= u_iterations; i++) {
                    float freq = i * (1.0 + u_distortion * 0.5);
                    v += sin(coord.x * freq + u_time);
                    v += sin(coord.y * freq + u_time);
                    
                    vec2 newCoord = coord;
                    newCoord.x = coord.x * cos(i) - coord.y * sin(i);
                    newCoord.y = coord.x * sin(i) + coord.y * cos(i);
                    coord = newCoord;
                }
                
                float r = sin(v * u_color_shift * 3.14 + u_time + u_hue_shift);
                float g = cos(v * u_color_shift * 3.14 + u_time + u_hue_shift);
                float b = sin(v * u_color_shift * 3.14 + u_time + 3.14/2.0 + u_hue_shift);
                
                vec3 col = vec3(r, g, b) * 0.5 + 0.5;
                
                // Apply saturation
                float gray = dot(col, vec3(0.299, 0.587, 0.114));
                col = mix(vec3(gray), col, u_saturation);
                
                f_color = vec4(col * u_brightness, 1.0);
            }
        '''

        # --- Shader 4: Cosmic Cloud ---
        shader_cloud = '''
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

            mat2 rotate2d(float angle){
                return mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
            }

            void main() {
                vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;
                
                // Apply rotation
                uv *= rotate2d(u_rotation);
                
                // Apply scrolling
                uv += vec2(u_scroll_x, u_scroll_y);
                
                float t = u_time * 0.1;
                
                vec3 finalColor = vec3(0.0);
                
                uv *= u_zoom * 2.0;

                for(float i=1.0; i<=u_iterations; i++){
                    uv += sin(uv.yx * (2.0 + u_distortion) + t + i) * 0.4;
                    uv *= rotate2d(t * 0.05 + i);
                    
                    float d = length(uv);
                    
                    float val = sin(d * (10.0 + u_distortion) - t * 2.0);
                    val = smoothstep(0.0, 0.8, abs(val));
                    
                    vec3 col = 0.5 + 0.5 * cos(t + vec3(0.0, 0.33, 0.67) * 6.28 + d * u_color_shift + u_hue_shift);
                    finalColor += col * val * (1.0/i);
                }
                
                // Apply saturation
                float gray = dot(finalColor, vec3(0.299, 0.587, 0.114));
                finalColor = mix(vec3(gray), finalColor, u_saturation);
                
                f_color = vec4(finalColor * u_brightness, 1.0);
            }
        '''

        # --- Shader 5: Breathing Mandala ---
        shader_mandala = '''
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

            mat2 rotate2d(float angle){
                return mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
            }

            void main() {
                vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;
                
                // Apply rotation
                uv *= rotate2d(u_rotation);
                
                // Apply scrolling
                uv += vec2(u_scroll_x, u_scroll_y);
                
                float breath = sin(u_time * 0.2) * 0.2 + 1.0;
                uv *= u_zoom * 2.0 * breath;
                
                float angle = atan(uv.y, uv.x);
                float dist = length(uv);
                
                vec3 finalColor = vec3(0.0);
                
                for(float i=0.0; i<u_iterations; i++){
                    float t_offset = u_time * 0.1 * (i + 1.0) * (1.0 + u_distortion * 0.5);
                    
                    float segments = 6.0 + i * 2.0;
                    float a = angle + t_offset;
                    a = mod(a, 6.28318 / segments) - (3.14159 / segments);
                    
                    vec2 p = vec2(cos(a), sin(a)) * dist;
                    
                    float d = length(p - vec2(0.5, 0.0));
                    d = sin(d * 20.0 - u_time * 2.0);
                    
                    vec3 col = 0.5 + 0.5 * cos(u_time * 0.5 + i + dist * u_color_shift + vec3(0,2,4) + u_hue_shift);
                    finalColor += col * smoothstep(0.1, 0.2, abs(d));
                }
                
                finalColor *= 1.0 - smoothstep(0.5, 1.5, dist);
                
                // Apply saturation
                float gray = dot(finalColor, vec3(0.299, 0.587, 0.114));
                finalColor = mix(vec3(gray), finalColor, u_saturation);
                
                f_color = vec4(finalColor * u_brightness, 1.0);
            }
        '''

        # --- Shader 6: Galactic Drift (Very Slow Evolution) ---
        shader_galaxy = '''
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

            mat2 rotate2d(float angle){
                return mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
            }

            // Hash function for star-like noise
            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            void main() {
                vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;
                
                // Apply rotation
                uv *= rotate2d(u_rotation);
                
                // Apply scrolling
                uv += vec2(u_scroll_x, u_scroll_y);
                
                // Ultra-slow galactic rotation
                float t = u_time * 0.02;
                uv *= rotate2d(t * 0.1);
                
                uv *= u_zoom;
                
                vec3 finalColor = vec3(0.0);
                
                // Spiral arms
                float angle = atan(uv.y, uv.x);
                float dist = length(uv);
                
                for(float i=0.0; i<u_iterations; i++){
                    // Each arm rotates slowly and independently
                    float armAngle = angle + dist * (2.0 + u_distortion) - t * (0.3 + i * 0.05);
                    float armDist = sin(armAngle * (3.0 + i)) * 0.3;
                    
                    float arm = smoothstep(0.2, 0.0, abs(dist - 0.5 - armDist));
                    
                    // Stars scattered along arms
                    vec2 starCoord = vec2(angle * 5.0 + i, dist * 10.0);
                    float stars = hash(floor(starCoord + t * 0.1));
                    stars = smoothstep(0.98, 1.0, stars) * smoothstep(0.3, 0.8, dist);
                    
                    // Color shifts over very long periods
                    vec3 armColor = 0.5 + 0.5 * cos(t * 0.3 + i * 2.0 + vec3(0, 2, 4) * u_color_shift + u_hue_shift);
                    vec3 starColor = vec3(1.0, 0.95, 0.9);
                    
                    finalColor += armColor * arm + starColor * stars;
                }
                
                // Central glow
                float core = exp(-dist * 3.0) * 0.5;
                finalColor += vec3(1.0, 0.8, 0.6) * core;
                
                // Apply saturation
                float gray = dot(finalColor, vec3(0.299, 0.587, 0.114));
                finalColor = mix(vec3(gray), finalColor, u_saturation);
                
                f_color = vec4(finalColor * u_brightness, 1.0);
            }
        '''

        # --- Shader 7: Tectonic Shift (Geological Timescale) ---
        shader_tectonic = '''
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

            mat2 rotate2d(float angle){
                return mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
            }

            float noise(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            float fbm(vec2 p) {
                float value = 0.0;
                float amplitude = 0.5;
                for(int i = 0; i < 6; i++) {
                    value += amplitude * noise(p);
                    p *= 2.0;
                    amplitude *= 0.5;
                }
                return value;
            }

            void main() {
                vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;
                
                // Apply rotation
                uv *= rotate2d(u_rotation);
                
                // Apply scrolling
                uv += vec2(u_scroll_x, u_scroll_y);
                
                // Extremely slow continental drift
                float t = u_time * 0.01;
                
                uv *= u_zoom * 2.0;
                
                vec3 finalColor = vec3(0.0);
                
                // Simulate tectonic plates
                for(float i=0.0; i<u_iterations; i++){
                    // Each plate moves in different direction
                    vec2 drift = vec2(sin(t * 0.2 + i), cos(t * 0.15 + i * 1.5)) * t * 0.05;
                    vec2 plateUV = uv + drift;
                    
                    // Domain warping for plate boundaries
                    plateUV += vec2(
                        fbm(plateUV * (2.0 + u_distortion) + t * 0.1),
                        fbm(plateUV * (2.0 + u_distortion) + t * 0.1 + 100.0)
                    ) * 0.3;
                    
                    float elevation = fbm(plateUV);
                    
                    // Different colors for different heights (ocean, land, mountains)
                    vec3 col;
                    if(elevation < 0.4) {
                        col = vec3(0.1, 0.2, 0.5); // Ocean
                    } else if(elevation < 0.6) {
                        col = vec3(0.2, 0.6, 0.3); // Plains
                    } else {
                        col = vec3(0.7, 0.7, 0.8); // Mountains
                    }
                    
                    // Color shifts over eons
                    col *= 0.5 + 0.5 * cos(t * 0.1 + elevation * u_color_shift + u_hue_shift);
                    
                    finalColor += col * (1.0 / (i + 1.0));
                }
                
                // Apply saturation
                float gray = dot(finalColor, vec3(0.299, 0.587, 0.114));
                finalColor = mix(vec3(gray), finalColor, u_saturation);
                
                f_color = vec4(finalColor * u_brightness, 1.0);
            }
        '''

        # --- Shader 8: Bioluminescent Tide (Lunar Cycle) ---
        shader_bio = '''
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

            mat2 rotate2d(float angle){
                return mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
            }

            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            void main() {
                vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;
                
                // Apply rotation
                uv *= rotate2d(u_rotation);
                
                // Apply scrolling
                uv += vec2(u_scroll_x, u_scroll_y);
                
                // Lunar tide cycle (very slow)
                float t = u_time * 0.05;
                float tide = sin(t * 0.1) * 0.3;
                
                uv *= u_zoom * 3.0;
                uv.y += tide;
                
                vec3 finalColor = vec3(0.0);
                
                // Bioluminescent organisms
                for(float i=0.0; i<u_iterations; i++){
                    vec2 cellCoord = uv * (5.0 + i * 2.0);
                    
                    // Slow drifting current
                    cellCoord.x += t * (0.1 + i * 0.05);
                    cellCoord.y += sin(cellCoord.x * 2.0 + t * 0.2) * (0.2 + u_distortion * 0.2);
                    
                    vec2 cellID = floor(cellCoord);
                    vec2 cellUV = fract(cellCoord) - 0.5;
                    
                    // Each organism pulses at its own rate
                    float h = hash(cellID + i);
                    float pulse = sin(t * (0.5 + h * 2.0) + h * 6.28) * 0.5 + 0.5;
                    pulse = smoothstep(0.3, 0.8, pulse);
                    
                    float dist = length(cellUV);
                    float organism = smoothstep(0.3, 0.1, dist) * pulse;
                    
                    // Cyan/blue/green bioluminescence
                    vec3 col = vec3(0.2, 0.8, 1.0);
                    col = 0.5 + 0.5 * cos(h * 6.28 + vec3(0, 2, 4) * u_color_shift + u_hue_shift);
                    
                    finalColor += col * organism * (1.0 / sqrt(i + 1.0));
                }
                
                // Underwater ambient
                vec3 ambient = vec3(0.0, 0.05, 0.15);
                finalColor += ambient;
                
                // Apply saturation
                float gray = dot(finalColor, vec3(0.299, 0.587, 0.114));
                finalColor = mix(vec3(gray), finalColor, u_saturation);
                
                f_color = vec4(finalColor * u_brightness, 1.0);
            }
        '''

        # --- Shader 9: Aurora Evolution (Seasonal) ---
        shader_aurora = '''
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

            mat2 rotate2d(float angle){
                return mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
            }

            float noise(vec2 p) {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }

            void main() {
                vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;
                
                // Apply rotation
                uv *= rotate2d(u_rotation);
                
                // Apply scrolling
                uv += vec2(u_scroll_x, u_scroll_y);
                
                // Seasonal variation (very slow)
                float t = u_time * 0.03;
                float season = sin(t * 0.05) * 0.5 + 0.5;
                
                uv *= u_zoom;
                
                vec3 finalColor = vec3(0.0);
                
                // Aurora curtains
                for(float i=0.0; i<u_iterations; i++){
                    // Vertical curtains that wave slowly
                    float x = uv.x * (3.0 + i);
                    float wave = sin(x + t * (0.3 + i * 0.1)) * (0.5 + u_distortion);
                    
                    float curtain = uv.y - wave;
                    curtain = abs(curtain);
                    
                    // Intensity varies with season and magnetic activity
                    float intensity = smoothstep(0.5, 0.0, curtain) * (0.5 + season * 0.5);
                    
                    // Aurora colors - greens, blues, reds
                    vec3 col = vec3(0.2, 1.0, 0.5); // Default green
                    
                    // Color shifts based on altitude (layers) and time
                    col = 0.5 + 0.5 * cos(t * 0.2 + i * 1.5 + vec3(0, 2, 4) * u_color_shift + u_hue_shift);
                    
                    // Add shimmer
                    float shimmer = noise(vec2(x * 20.0, t * 2.0 + i));
                    intensity *= 0.7 + shimmer * 0.3;
                    
                    finalColor += col * intensity;
                }
                
                // Dark sky background
                finalColor += vec3(0.01, 0.01, 0.03);
                
                // Stars (subtle)
                float stars = noise(uv * 100.0);
                stars = smoothstep(0.99, 1.0, stars);
                finalColor += stars * 0.3;
                
                // Apply saturation
                float gray = dot(finalColor, vec3(0.299, 0.587, 0.114));
                finalColor = mix(vec3(gray), finalColor, u_saturation);
                
                f_color = vec4(finalColor * u_brightness, 1.0);
            }
        '''

        # --- Shader 10: Crystalline Growth (Slow Formation) ---
        shader_crystal = '''
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

            mat2 rotate2d(float angle){
                return mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
            }

            void main() {
                vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution.xy) / u_resolution.y;
                
                // Apply rotation
                uv *= rotate2d(u_rotation);
                
                // Apply scrolling
                uv += vec2(u_scroll_x, u_scroll_y);
                
                // Slow crystal growth
                float t = u_time * 0.02;
                float growth = smoothstep(0.0, 10.0, t) * (1.0 + sin(t * 0.1) * 0.2);
                
                uv *= u_zoom * (1.0 + growth * 0.5);
                
                vec3 finalColor = vec3(0.0);
                
                // Hexagonal crystal structure
                float angle = atan(uv.y, uv.x);
                float dist = length(uv);
                
                for(float i=0.0; i<u_iterations; i++){
                    // 6-fold symmetry
                    float a = angle + t * 0.05 * (i + 1.0);
                    a = mod(a, 6.28318 / 6.0);
                    
                    // Growing crystal faces
                    float face = abs(sin(a * 3.0 + t * 0.1));
                    float layer = sin(dist * (10.0 + i * 3.0) - t * 0.3 + face * u_distortion) * 0.5 + 0.5;
                    
                    // Each layer forms at different times
                    float formation = smoothstep(i * 0.5, i * 0.5 + 2.0, t);
                    layer *= formation;
                    
                    // Refraction-like colors
                    vec3 col = vec3(0.5, 0.8, 1.0);
                    col = 0.5 + 0.5 * cos(dist * u_color_shift + i + vec3(0, 2, 4) + u_hue_shift);
                    
                    finalColor += col * layer * smoothstep(1.0, 0.5, dist);
                }
                
                // Central seed crystal
                float core = exp(-dist * 5.0);
                finalColor += vec3(1.0, 1.0, 0.9) * core;
                
                // Apply saturation
                float gray = dot(finalColor, vec3(0.299, 0.587, 0.114));
                finalColor = mix(vec3(gray), finalColor, u_saturation);
                
                f_color = vec4(finalColor * u_brightness, 1.0);
            }
        '''

        self.shader_sources = [
            shader_fractal, shader_grid, shader_plasma, shader_cloud, shader_mandala,
            shader_galaxy, shader_tectonic, shader_bio, shader_aurora, shader_crystal
        ]
        self.shader_names = [
            "Fractal", "Grid", "Plasma", "Cloud", "Mandala",
            "Galaxy", "Tectonic", "Bioluminescent", "Aurora", "Crystal"
        ]
        self.programs = []
        self.vaos = []

        # ------------------------------------------------------------------
        # GEOMETRY & COMPILATION
        # ------------------------------------------------------------------
        
        vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0,
        ], dtype='f4')
        self.vbo = self.ctx.buffer(vertices)

        # Compile all shaders
        for source in self.shader_sources:
            prog = self.ctx.program(vertex_shader=self.vertex_shader, fragment_shader=source)
            vao = self.ctx.vertex_array(prog, [(self.vbo, '2f', 'in_vert')])
            self.programs.append(prog)
            self.vaos.append(vao)

        self.current_shader_idx = 0
        self.prog = self.programs[0]
        self.vao = self.vaos[0]

        # ------------------------------------------------------------------
        # FRAMEBUFFER SETUP
        # ------------------------------------------------------------------
        self.fbo_texture = self.ctx.texture((self.width, self.height), 3)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.fbo_texture])
        
        # Initialize defaults
        self.update_all_uniforms({
            'u_resolution': (self.width, self.height),
            'u_zoom': 1.5,
            'u_distortion': 0.5,
            'u_iterations': 4.0,
            'u_color_shift': 1.0,
            'u_brightness': 1.0,
            'u_hue_shift': 0.0,
            'u_saturation': 1.0,
            'u_scroll_x': 0.0,
            'u_scroll_y': 0.0,
            'u_rotation': 0.0
        })

    def toggle_shader(self):
        """Switches to the next shader in the list."""
        self.current_shader_idx = (self.current_shader_idx + 1) % len(self.programs)
        self.prog = self.programs[self.current_shader_idx]
        self.vao = self.vaos[self.current_shader_idx]
        print(f"Switched to Shader: {self.shader_names[self.current_shader_idx]}")

    def set_shader(self, index):
        """Set shader by index (0-4)."""
        if 0 <= index < len(self.programs):
            self.current_shader_idx = index
            self.prog = self.programs[index]
            self.vao = self.vaos[index]

    def update_all_uniforms(self, params_dict):
        """Push all current params to active shader only."""
        for name, value in params_dict.items():
            try:
                if name in self.prog:
                    self.prog[name].value = value
            except Exception:
                pass

    def set_uniform(self, name, value):
        """Set uniform on active shader only."""
        try:
            if name in self.prog:
                self.prog[name].value = value
        except Exception:
            pass

    def render(self, time_val, params_dict):
        """Render current shader and return frame."""
        self.ctx.clear(0.0, 0.0, 0.0)
        
        # Update all uniforms including time on active shader
        for name, value in params_dict.items():
            self.set_uniform(name, value)
        self.set_uniform('u_time', time_val)
        
        self.fbo.use()
        self.vao.render(moderngl.TRIANGLE_STRIP)
        
        raw_data = self.fbo.read(components=3, alignment=1)
        image = np.frombuffer(raw_data, dtype=np.uint8)
        image = image.reshape((self.height, self.width, 3))
        image = cv2.flip(image, 0) 
        
        return image


# ------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------
if __name__ == "__main__":
    WIDTH, HEIGHT = 1280, 720
    viz = ShaderVisualizer(width=WIDTH, height=HEIGHT)
    
    start_time = time.time()
    
    # Tunable parameters - connect these to DearPyGUI controls
    params = {
        # Core shader parameters
        "u_zoom": 1.5,
        "u_distortion": 0.2,
        "u_iterations": 4.0,
        "u_color_shift": 1.0,
        "u_brightness": 1.0,
        
        # Extended parameters
        "u_hue_shift": 0.0,
        "u_saturation": 1.0,
        "u_scroll_x": 0.0,
        "u_scroll_y": 0.0,
        "u_rotation": 0.0,
        
        # Static
        "u_resolution": (WIDTH, HEIGHT)
    }

    print("=" * 70)
    print("SHADER VISUALIZER - 10 SHADERS WITH LONG EVOLUTION PERIODS")
    print("=" * 70)
    print("\nShaders (1-0 keys):")
    print("  1. Fractal      - Fast psychedelic patterns")
    print("  2. Grid         - Hypnotic geometric layers")
    print("  3. Plasma       - Liquid wave interference")
    print("  4. Cloud        - Slow cosmic nebula (10x slower)")
    print("  5. Mandala      - Breathing sacred geometry")
    print("  6. Galaxy       - Galactic drift (50x slower)")
    print("  7. Tectonic     - Continental drift (100x slower)")
    print("  8. Bioluminescent - Lunar tide cycle (20x slower)")
    print("  9. Aurora       - Seasonal aurora (30x slower)")
    print("  0. Crystal      - Crystalline growth (50x slower)")
    print("\nTunable Parameters for DearPyGUI:")
    print("  u_zoom         : Scale/zoom level (0.1 - 5.0)")
    print("  u_distortion   : Distortion amount (0.0 - 2.0)")
    print("  u_iterations   : Complexity/layers (1.0 - 10.0)")
    print("  u_color_shift  : Color frequency (0.0 - 2.0)")
    print("  u_brightness   : Overall brightness (0.0 - 2.0)")
    print("  u_hue_shift    : Hue rotation (0.0 - 6.28)")
    print("  u_saturation   : Color saturation (0.0 - 2.0)")
    print("  u_scroll_x/y   : Position offset (-5.0 - 5.0)")
    print("  u_rotation     : Rotation angle (0.0 - 6.28)")
    print("\nKeyboard Controls:")
    print(" [1-9,0]: Select shader directly")
    print(" [T]:     Toggle to next shader")
    print(" [R]:     Reset parameters")
    print(" [ESC]:   Quit")
    print("=" * 70)

    while True:
        current_time = time.time() - start_time
        
        # Render current shader with all params
        frame = viz.render(current_time, params)
        
        # Display info
        shader_name = viz.shader_names[viz.current_shader_idx]
        text = f"Shader: {shader_name} ({viz.current_shader_idx+1}/10)"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Show parameter values
        y_offset = 60
        cv2.putText(frame, f"Zoom: {params['u_zoom']:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Dist: {params['u_distortion']:.2f}", (10, y_offset+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Iter: {params['u_iterations']:.0f}", (10, y_offset+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Shader Visualizer", frame)
        
        # Handle input
        key = cv2.waitKey(16) & 0xFF
        if key == 27:  # ESC
            break
        
        # Shader selection
        if key == ord('1'):
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
        elif key == ord('t') or key == ord('T'):
            viz.toggle_shader()
        elif key == ord('r') or key == ord('R'):
            # Reset parameters
            params.update({
                "u_zoom": 1.5,
                "u_distortion": 0.2,
                "u_iterations": 1.0,
                "u_color_shift": 1.0,
                "u_brightness": 1.0,
                "u_hue_shift": 0.0,
                "u_saturation": 1.0,
                "u_scroll_x": 0.0,
                "u_scroll_y": 0.0,
                "u_rotation": 0.0
            })
            print("Parameters reset")

    cv2.destroyAllWindows()