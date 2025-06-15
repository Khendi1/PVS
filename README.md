# video_synth

This is an interactive video synthesis tool with many tunable parameters. Most parameters can be modulated by oscillators.

## Requirements
- python3
- packages from requirements.txt
- webcam (for now; video looping planned)

## Features and parameters:
- Feedback (nothing fancy for now, only cv2.addWeighted())
- HSV control
- Contrast/brightness control
- locally save and recall patches (untested in awhile, likely broken)
- Frame pan, tilt, zoom
- Perlin noise generator
- Polar Coordinate transform center and intensity
- fix glitch
- Oscillator bank
    - tunable freq, amp, phase
    - supports 4 basic wave forms
- shape generator
    - line & fill weight, hsv, opacity
    - size, position control
    - multiplication across x and y axis
    - multiplication pitch across x and y axis
    - rotation

## Roadmap:
- fully integrate every parameter for oscillators. CUrrently broken:
    - after linking variables anf then selecting oscillator linked variable None, behavior continues
    - shape rotation broken
    - more
- button to pan rotate 90deg
- investigate bug in glitch
- use perlin noise, klaman to filter/resist hue change
- command line arg for more oscillators
- command line arg for video input with option to loop video fileu
- investigate ways to reduce flicker and strobing effects
- VCO emulation 
- integrate perlin noise into oscillators wave form generator
- automate generation of gui slider on Param creation