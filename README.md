# video_synth

Analog video synthesizer modules are expensive, CRTs are bulky and liquid light shows are messy. If you want to play around with live video effects without patch cables, dyes or expensive software, try out the Python Video Synthesizer (PVS).  Or augment your existing setup for as little as $0.00!

PVS is designed for use with MIDI controllers. Turning knobs and pushing faders is more engaging than using a mouse, though a (crappy) GUI is provided. 

There are many tunable parameters (~150 at the time of writing). Most parameters can be modulated by oscillators, including the oscillator parameters themselves (i.e. oscillator frequency, amplitude, phase, vertical shift).

This suite plays well with lots of other tools and hardware, so take a look at the [Optional Off-The-Shelf Hardware](#optional-off-the-shelf-hardware), [Optional DIY Hardware](#optional-diy-hardware), and [Optional Custom PCBs](#optional-custom-pcbs) sections.


## Features and Parameters:
- Feedback & Filtering:
    - Alpha blend: blends raw frame with modified frame
    - Temporal filter: blends current alpha-blend frame with previous alpha-blend frame to reduce strobing effects
    - Frame buffer: store and average temporal-filter frames in variable length frame buffer
- 2 Source mixer:
    - alpha blend mode
    - luma keying with white/black selection
    - chroma keying
- Color control:
    - hue, saturation, value
    - contrast, brightness
    - posterize
    - solarize
- Pixel manipulations:
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
- Shape generator
    - show circles, squares, triangles, short rays 
    - shape line weight & fill color parameters (hsv, opacity)
    - shape size, position, and rotation control
    - tiling across x and y axis
    - tiling pitch across x and y axis
- Pattern generator
    - attempts to emulate oscilator effects used analog video synthesis 
    - creates moving BRG bars, waves, concentric circles
- metaballs
    - attempts to recreate a classic animation[] technique
- reaction diffusion simulator
- plasma generator
- locally save and recall patches (currently broken after imlementing params class)

## Background and Inspiration

My exploration of video manipulation was inspired by [this video](https://www.youtube.com/watch?v=D3eHKI0nvKA), which was graciously provided by the algorithm. In describing his live image manipulator mechanical sculpture masterpeice, Dave Blair describes why his machine must use an expensive field-monitor commonly used in movie production, as it offers knobs to manipulate hue, saturation and value. While this requirement makes sense for his application, I was inspired to explore a purely code-based solution, with the intention to solder up some encoders. I have since pivoted to using off-the-shelf controllers to emulate effects normally acheived through analog video synthesis modules and mixers (feedback, oscillators, [composite analog sync modulation](https://www.youtube.com/watch?v=YlLN_H3Z8Gc)), as well as early digital animation techniques (perlin noise, fractal noise, [metaballs](https://steve.hollasch.net/cgindex/misc/metaballs.html), plasma, moire, etc.).

## Requirements
- python3
- packages from ```requirements.txt```
- webcam

## Optional Off-The-Shelf Hardware
- Midi controllers (see [Control](#control) section)
- HDMI capture device (hdmi pass-thru can be a useful feature)
- HDMI 1-in-4-out
- HDMI to composite converter
- HDMI to VGA converter
- Composite to HDMI converter 
- Composite capture device
- VGA to HDMI converter
- VGA capture device

## Optional DIY Hardware
- GBS Video Feedback Synth (also see notes from Gleix for bend)
- 2-in-1 out composite switcher
- Klompf Dirty Video Mixer
- VC9900 Glitch Box w/ GBSControl

## Optional Custom PCBs
- CHA/V by Jonas Bers
- Archer Enhancer by lightmusicvisuals
- PixelSlasher by CTXz
- recurBOY by cyberboy666
- _rupture_ by cyberboy666
- sync_ope by cyberboy666

## Control
The app can be controlled through a (crappy) GUI or midi controller. The program is currently configured to use a cheap AKAI MIDI Mix or MVAVE SMC Mixer, but there are provisions to configure a new controller with relative ease. To implement a new controller, see the required functions you must expose in the example ```SMC_Mixer()``` or ```MidiMix()``` classes in ```midi.py```. Most work involves the mapping.

The current configuration will be superceded by a class/subclass structure.

## My Current Configuration
![Video Synthesizer Diagram](documentation/diagram.svg)

