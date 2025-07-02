import cv2
import numpy as np
from noise import pnoise2

# Constants
WIDTH, HEIGHT = 640, 480
PERLIN_SCALE = 0.01
feedback = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

def nothing(x): pass

# UI setup
cv2.namedWindow("Controls")
cv2.resizeWindow("Controls", 600, 800)

# Warp mode: 0 = sine, 1 = perlin, 2 = polar
cv2.createTrackbar("Warp Mode", "Controls", 0, 4, nothing)

# Perlin/sine parameters
cv2.createTrackbar("X Amplitude", "Controls", 20, 100, nothing)
cv2.createTrackbar("Y Amplitude", "Controls", 20, 100, nothing)
cv2.createTrackbar("X Freq", "Controls", 10, 100, nothing)
cv2.createTrackbar("Y Freq", "Controls", 10, 100, nothing)

# Polar parameters
cv2.createTrackbar("Polar Angle Amt", "Controls", 30, 180, nothing)
cv2.createTrackbar("Polar Radius Amt", "Controls", 30, 180, nothing)

# Common controls
cv2.createTrackbar("Speed", "Controls", 10, 100, nothing)
cv2.createTrackbar("Blend", "Controls", 90, 100, nothing)

# Fractal Noise Toggle and Parameters
cv2.createTrackbar("Fractal Noise", "Controls", 1, 1, nothing)
cv2.createTrackbar("Octaves", "Controls", 4, 8, nothing)
cv2.createTrackbar("Gain", "Controls", 50, 100, nothing)        # 0.0 - 1.0
cv2.createTrackbar("Lacunarity", "Controls", 20, 40, nothing)   # 1.0 - 4.0

# Perlin noise field
def generate_perlin_flow(t, amp_x, amp_y, freq_x, freq_y):
    fx = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    fy = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            nx = x * freq_x * PERLIN_SCALE
            ny = y * freq_y * PERLIN_SCALE
            fx[y, x] = amp_x * pnoise2(nx, ny, base=int(t))
            fy[y, x] = amp_y * pnoise2(nx + 1000, ny + 1000, base=int(t))
    return fx, fy

# Fractal Perlin Noise Generator
def generate_fractal_flow(t, amp_x, amp_y, freq_x, freq_y, octaves, gain, lacunarity):
    fx = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    fy = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

    for y in range(HEIGHT):
        for x in range(WIDTH):
            nx = x * freq_x * PERLIN_SCALE
            ny = y * freq_y * PERLIN_SCALE

            noise_x = 0.0
            noise_y = 0.0
            amplitude = 1.0
            frequency = 1.0

            for _ in range(octaves):
                noise_x += amplitude * pnoise2(nx * frequency, ny * frequency, base=int(t))
                noise_y += amplitude * pnoise2((nx + 1000) * frequency, (ny + 1000) * frequency, base=int(t))
                amplitude *= gain
                frequency *= lacunarity

            fx[y, x] = amp_x * noise_x
            fy[y, x] = amp_y * noise_y
    return fx, fy

# Polar warp function
def polar_warp(img, t, angle_amt, radius_amt, speed):
    cx, cy = WIDTH / 2, HEIGHT / 2
    y, x = np.indices((HEIGHT, WIDTH), dtype=np.float32)
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2)
    a = np.arctan2(dy, dx)

    # Modify radius and angle
    r_mod = r + np.sin(a * 5 + t * speed * 2) * radius_amt
    a_mod = a + np.cos(r * 0.02 + t * speed * 2) * (angle_amt * np.pi / 180)

    # Back to Cartesian
    map_x = (r_mod * np.cos(a_mod) + cx).astype(np.float32)
    map_y = (r_mod * np.sin(a_mod) + cy).astype(np.float32)

    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def warp(img, t, mode, amp_x, amp_y, freq_x, freq_y, angle_amt, radius_amt, speed, use_fractal, octaves, gain, lacunarity):
    if mode == 0: # Sine warp
        fx = np.sin(np.linspace(0, np.pi * 2, WIDTH)[None, :] + t) * amp_x
        fy = np.cos(np.linspace(0, np.pi * 2, HEIGHT)[:, None] + t) * amp_y
        map_x, map_y = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
        map_x = (map_x + fx).astype(np.float32)
        map_y = (map_y + fy).astype(np.float32)
        return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    elif mode == 1:
        if use_fractal:
            fx, fy = generate_fractal_flow(t, amp_x, amp_y, freq_x, freq_y, octaves, gain, lacunarity)
        else:
            fx, fy = generate_fractal_flow(t, amp_x, amp_y, freq_x, freq_y, 1, gain, lacunarity)  # fallback: single octave
        map_x, map_y = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
        map_x = (map_x + fx).astype(np.float32)
        map_y = (map_y + fy).astype(np.float32)
        return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    elif mode == 2:
        return polar_warp(img, t, angle_amt, radius_amt, speed)
    
    elif mode == 3:
        # Fractal Perlin warp
        fx, fy = generate_perlin_flow(t, amp_x, amp_y, freq_x, freq_y)
        map_x, map_y = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
        map_x = (map_x + fx).astype(np.float32)
        map_y = (map_y + fy).astype(np.float32)
        return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

cap = cv2.VideoCapture(0)
t = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Get control values
    mode = cv2.getTrackbarPos("Warp Mode", "Controls")
    amp_x = cv2.getTrackbarPos("X Amplitude", "Controls")
    amp_y = cv2.getTrackbarPos("Y Amplitude", "Controls")
    freq_x = cv2.getTrackbarPos("X Freq", "Controls") / 10.0
    freq_y = cv2.getTrackbarPos("Y Freq", "Controls") / 10.0
    angle_amt = cv2.getTrackbarPos("Polar Angle Amt", "Controls")
    radius_amt = cv2.getTrackbarPos("Polar Radius Amt", "Controls")
    speed = cv2.getTrackbarPos("Speed", "Controls") / 100.0
    blend = cv2.getTrackbarPos("Blend", "Controls") / 100.0

    use_fractal = cv2.getTrackbarPos("Fractal Noise", "Controls")
    octaves = cv2.getTrackbarPos("Octaves", "Controls")
    gain = cv2.getTrackbarPos("Gain", "Controls") / 100.0
    lacunarity = cv2.getTrackbarPos("Lacunarity", "Controls") / 10.0

    warped = warp(frame, t, mode, amp_x, amp_y, freq_x, freq_y, angle_amt, radius_amt, speed,
                  use_fractal, max(1, octaves), max(0.01, gain), max(0.01, lacunarity))
    
    feedback = cv2.addWeighted(frame, 1 - blend, warped, blend, 0)
    
    cv2.imshow("Feedback Synth Warp", feedback)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    t += 0.01

cap.release()
cv2.destroyAllWindows()