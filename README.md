# video_synth

This is an interactive video synthesis tool with many tunable parameters. Most parameters can be modulated by oscillators, including the oscillator parameters themselves.

## Requirements
- python3
- packages from requirements.txt
- webcam (for now; video looping planned)

## Control

The app can be controlled through a (crappy) GUI or midi controller. The program is currently configured to use a cheap MVAVE SMC Mixer, but there are provisions to configure a new controller with relative ease. To reimplement a new controller, see the required functions you must expose in the example SMC_Mixer() class in midi.py

## Parameters:
- Feedback: blends stream frame with previous frame
- Temporal filter: blends current generated feedback frame with previous feedback frame to reduce strobing effects
- HSV control
- Contrast/brightness control
- Frame pan, tilt, zoom
- Polar Coordinate transform: center position and radius size
- glitch generator (size, quantity, ...)
- Various blur modes
- Various noise modes
- x/y sync modulation emulator
    - sync freq
    - sync amplitude
    - sync speed (need better word)
- Oscillator bank
    - tunable freq, amp, phase
    - supports 4 basic wave forms
    - can be linked to any parameter in the ParamsTable
    - can be linked to other oscillator parameters
- shape generator
    - line & fill weight, hsv, opacity
    - shape size, position, and rotation control
    - multiplication across x and y axis
    - multiplication pitch across x and y axis
- Perlin noise generator (not currently used)
- locally save and recall patches (untested in awhile, likely broken)

## Known issues
- buttons are generally broken. 
    - Saving/loading patches is broken
    - Randomizing param values are broken
     
## Roadmap:
- button to pan rotate 90deg
- investigate bug in glitch
- use perlin noise, klaman to filter/resist hue change
- command line arg for more oscillators
- command line arg for video input with option to loop video fileu
- integrate perlin noise into oscillators wave form generator
- automate generation of gui slider on Param creation
- ASDR envelope generator
- sync osc phase
- rotate shape canvas
- cleanup args for TrackbarCallback (second arg is redundant, can be gathered from first arg)
- cleanup args Trackbar args (first arg is redundant, can be gathered from others)


## unimplemented effects

warp

