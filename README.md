# Python Video Synthesizer

Analog video synth modules are expensive, CRTs are bulky and liquid light shows are messy. If you want to play around with live video effects without patch cables, dyes or expensive software, try out the Python Video Synthesizer (PVS).  Or augment your existing setup for as little as $0.00! 

PVS is designed for use with MIDI controllers. Turning knobs and pushing faders is more engaging than using a mouse, though a (crappy) GUI is provided. 

There are many tunable parameters (~250 at the time of writing). Most parameters can be modulated by oscillators, including the oscillator parameters themselves (i.e. oscillator frequency, amplitude, phase, vertical shift).

This suite plays well with lots of other tools and hardware. Note that any additional video capture devices, webcams, mixers, etc are **optional!** I have spent hours playing with this toy with nothing but a cheap laptop and its built-in camera, though it will even work without a webcam.

## Requirements
- python3
- packages from ```requirements.txt```
- webcam

## Software Setup
1. Clone this repository

2. from the top level directory, create a virtual environment:

    - ```python -m venv <YOUR-VENV-NAME>```

3. activate your virtual environment

    - Windows: ```<YOUR-VENV_NAME>/Scripts/activate```

    - Linux: ```source <YOUR-VENV_NAME>/bin/activate```

4. install packages from requirements.txt.

    - ```pip install -r requirements.txt```

5. *OPTIONAL*: configure hardware; plug in MIDI devices and capture devices, See [Hardware Setup](#hardware-setup) for additional details.

6. still from the top level, launch the program:

    - ```python video_synth```

## Hardware setup

Hardware video devices (USB capture devices, webcams, etc.) and MIDI controllers are automatically identified upon program execution. They are not re-identified or recovered through program execution (boo), so restart the program if you ever unplug equipment.

My personal MIDI devices have been mapped, but you will need to do this for your model of controller. See the [Control](#control) section to configure your MIDI device. 

Find the 'Mixer' GUI panel to select among different video devices. Currently devices are identified by their USB enumeration value, i.e. the order that they are plugged in.

## Features and Parameters:
- Locally save and recall patches
- 2 Source mixer:
    - alpha blending
    - luma keying with white/black selection
    - chroma keying
    - supports live usb devices like capture cards and webcams, saved image and video files, and some custom animations/simulations
- Animated Input Sources:
    - Metaballs
    - Reaction diffusion
    - Plasma generator
    - Moire pattern generator
    - Shader controller
- Feedback & Filtering:
    - Alpha blend: blends opacity of raw captured frame with  previous modified frame
    - Temporal filter: blends current alpha-blend frame with previous alpha-blend frame to reduce strobing effects
    - Frame buffer: store and average temporal-filter frames in variable length frame buffer
- LFO menu: 
    - each parameter has an optional LFO
    - each LFO has a set of parameters (i.e. frequency, amplitude, etc.), that themselves have optional LFOs, with the depth only limited by my poor gui logic and your available screen space
- Effects Manager:
    - source 1, source 2, and the mixed output all have individual effect controls
    - effects can be sequenced individually
- Effects
    - Color control
    - Pixel manipulations
    - Glitches
    - Reflection/mirroring
    - Frame pan, tilt, zoom
    - Polar coordinate transform
    - Sync modulation emulator
    - Shape generator
    - Pattern generator
    - Warp effect generator

#### Parameters
The ```Param``` class is used to manage all control variables. Each ```Param``` is a number with a minimum and maximum values (>=2), and belongs to. 

```Param```s are primarily used by the effects classes, but are also used in the Mixer and LFO bank. Essentially any single value that a user can alter should be a ```Param```.


#### Effects Classes and Effects Manager
For effect modularity and gui simplicity, each related set of effects are placed into an Effects Class.
Each effect class is derived from the ```EffectBase``` parent class.
Each effect class should take the ```ParamsTable``` and ```ButtonsTable``` as arugments, add its own params, and implement its own ```create_gui_panel``` method. Until this is automated, the ```create_gui_panel``` must be manually called in ```gui.py/create_interface()```***.

All effects classes are stored, initialized and managed by the ```EffectsManager``` facade class. This simplifies dependencies, arguments, and effect sequencing. Feedback is currently ommitted from effect sequencing until further experimentation. 

## Control
The app can be controlled through a (crappy) GUI or midi controller. The program is currently configured to use a cheap AKAI MIDI Mix or MVAVE SMC Mixer, but there are provisions to configure a new controller with relative ease. To implement a new controller, see the required functions you must expose in the example ```SMC_Mixer()``` or ```MidiMix()``` classes in ```midi.py```. Most work involves the mapping of CC values to parameters.

Any new midi device class must be named after its parsed device name.
This can be found simply by running the program and finding the detected device name.
For example, the program may find an input device named "MIDI Mix 1". Remove the port number (resulting in "MIDI Mix"), and name your new class after it.
Add the new class ```.___name___``` attribute to the ```CONTROLLER_NAMES``` list, and the program should now ID your device.


## My Current Configuration
* TODO: include VGA capture device in sources
* TODO: include CHA/V, link to VGA capture source
* TODO: add posterize, solarize, to effects
* note that the Unmixed HDMI Display is not the fully mixed video. It is essentially a placeholder to indicate that there is another HDMI output available.

![Video Synthesizer Diagram](documentation/diagram.svg)

## Background and Inspiration

This program was inspired by [this video](https://www.youtube.com/watch?v=D3eHKI0nvKA), which was graciously provided by the algorithm despite lacking any prior exposure to video synthesis. In describing his kinetic video feedback synthesizer, Dave Blair explains why his machine use expensive field-monitors and their analog knobs to manipulate image properties. I wanted to acheive similar effects without such expensive equipment, so I was inspired to explore a purely code-based solution. The original intention was to design and program a PCB with an HID-capable MCU + encoders and interface it to a basic openCV program.

I have since pivoted to using off-the-shelf MIDI controllers to emulate effects normally acheived through analog video synthesis modules and mixers, and added various live, tunable simulations, optical art generators, and other visually interesting algorithms.