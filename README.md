# video_synth

Analog video synthesizer modules are expensive, CRTs are bulky and liquid light shows are messy. If you want to play around with live video effects without patch cables, dyes or expensive software, try out the Python Video Synthesizer (PVS).  Or augment your existing setup for as little as $0.00!

PVS is designed for use with MIDI controllers. Turning knobs and pushing faders is more engaging than using a mouse, though a (crappy) GUI is provided. 

There are many tunable parameters (~150 at the time of writing). Most parameters can be modulated by oscillators, including the oscillator parameters themselves (i.e. oscillator frequency, amplitude, phase, vertical shift).

## Background and Inspiration
Despite my limited knowledge or exposure live video manipulation, the YouTube algorithm graciously provided me with this video. In describing his live image manipulator mechanical sculpture masterpeice, Dave Blair describes why his machine must use an expensive field-monitor commonly used in movie production, as it offers knobs to manipulate hue, saturation and value. While this requirement makes sense for his application, I was inspired to explore a purely code-based solution. I have since tried to emulate effects that can be acheived through analog video synthesis modules and mixers (feedback, oscillators, [composite analog sync modulation](https://www.youtube.com/watch?v=YlLN_H3Z8Gc)), as well as early digital animation techniques (perlin noise, fractal noise, [metaballs](https://steve.hollasch.net/cgindex/misc/metaballs.html))

## Requirements
- python3
- packages from ```requirements.txt```
- webcam (video looping on roadmap)

## Control
The app can be controlled through a (crappy) GUI or midi controller. The program is currently configured to use a cheap AKAI MIDI Mix or MVAVE SMC Mixer, but there are provisions to configure a new controller with relative ease. To implement a new controller, see the required functions you must expose in the example ```SMC_Mixer()``` or ```MidiMix()``` classes in ```midi.py```. Most work involves the mapping 

## Parameters:
- Feedback: blends stream frame with previous frame
- Temporal filter: blends current generated feedback frame with previous feedback frame to reduce strobing effects
- Color control:
    - hue, saturation, value
    - contrast, brightness
    - posterize
    - solarize
- pixel manipulations:
    - sharpen insensity
    - various blur modes
    - various noise modes
    - glitch generator (size, quantity, ...)
- Frame pan, tilt, zoom
- Polar Coordinate transform: center position and radius size
- x/y sync modulation emulator
    - sync freq
    - sync amplitude
    - sync speed (need better word)
- Oscillator bank
    - tunable frequency, amplitude, phase, vertical offset
    - supports 4 basic wave forms
    - supports a perlin noise wave form for more dynamic modulations
    - can be linked to any parameter in the ParamsTable
    - can be linked to other oscillator parameters
- shape generator
    - show circles, squares, triangles, short rays 
    - shape line weight & fill color parameters (hsv, opacity)
    - shape size, position, and rotation control
    - tiling across x and y axis
    - tiling pitch across x and y axis
- pattern generator
    - attempts to emulate oscilator effects used analog video synthesis 
    - creates moving BRG bars, waves, concentric circles
- metaballs
    - attempts to recreate a classic animation[] technique
- locally save and recall patches (currently broken after imlementing params class)

## Known issues
- Buttons class is not fully implemented or tested
- save.py has not been updated to work with Params or ParamsTable

## Roadmap:
1. implement save/recall feature
2. map load patch buttons (fwd, random, back) to midi controller 
3. implement feature to load, play, and loop video
4. dial in parameters on pattern generator (wave and radial are kinda ugly but have potential)
5. implement plasma generator
6. map plasma generator controls to SMC mixer
7. implement control page function on SMC Mixer

