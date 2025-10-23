# Python Video Synthesizer

Analog video synthesizer modules are expensive, CRTs are bulky and liquid light shows are messy. If you want to play around with live video effects without patch cables, dyes or expensive software, try out the Python Video Synthesizer (PVS).  Or augment your existing setup for as little as $0.00!

PVS is designed for use with MIDI controllers. Turning knobs and pushing faders is more engaging than using a mouse, though a (crappy) GUI is provided. 

There are many tunable parameters (~150 at the time of writing). Most parameters can be modulated by oscillators, including the oscillator parameters themselves (i.e. oscillator frequency, amplitude, phase, vertical shift).

This suite plays well with lots of other tools and hardware, so take a look at the [Optional Off-The-Shelf Hardware](#optional-off-the-shelf-hardware), [Optional DIY Hardware](#optional-diy-hardware), and [Optional Custom PCBs](#optional-custom-pcbs) sections. Note that the tools in these sections are all **optional!** I have spent hours playing with this toy with nothing but a cheap laptop and its built-in camera.

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

5. *OPTIONAL*: configure hardware; plug in MIDI devices and capture devices, adjust MixSource enum in mix.py accordingly. See [Hardware Setup](#hardware-setup) for additional details.

6. still from the top level, launch the program:

    - ```python video_synth```

## Hardware setup

Configuring your MIDI hardware, cameras, and capture devices is currently a weak point, and will be revisted for overhaul. 

Cv2 camera/capture devices are identified upon program execution, but MIDI devices are not.

See the [Control](#control) section for guidance on configuring your MIDI device.


## Features and Parameters:
- Locally save and recall patches
    - if using USB hardware, this requires you to use the same hardware configuration
- 2 Source mixer:
    - alpha blend mode
    - luma keying with white/black selection
    - chroma keying
    - supports live, saved, and animated video sources
- Live Video Input Sources
    - supports up to 5 capture devices that openCV can recognize as a webcam. USB webcam and USB HDMI/Composite/VGA capture devices have all tested successfully.
- Video and Image File Input Source
    - support for looping video clips or displaying static images
- Animated Input Sources:
    - Metaballs
    - Reaction diffusion simulator
    - Plasma generator
    - Moire pattern generator
    - Shader controller
- Feedback & Filtering:
    - Alpha blend: blends raw frame (dry) with modified frame (wet)
    - Temporal filter: blends current alpha-blend frame with previous alpha-blend frame to reduce strobing effects
    - Frame buffer: store and average temporal-filter frames in variable length frame buffer
- Color control:
    - hue, saturation, value
    - contrast, brightness
    - posterize
    - solarize
    - hue invert angle, hue invert strength
    - value threshold, hue shift for value threshold
- Pixel manipulations:
    - sharpen insensity
    - various blur modes
    - various noise modes
- glitch generator 
    - splitscan
    - glitch (size, quantity, ...)
    - ...
    - ...
- Various frame reflection modes and parameters
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
    - attempts to emulate oscilator effects used in analog video synthesis 
    - creates moving BRG bars, waves, concentric circles, etc
- Warp effect generator
- Shader control

## Pending Features & Issue Tracking

This project is meant to be a fun, relaxing, creative exploration, not a demonstration of proper project management techniques. Instead of tying up precious creative time with Github issue tracking, task tags (i.e. TODO, BUG, etc.) are instead stored inline. The VSCode TODO Tree extension is useful for displaying these tags using various views.

## Program Archetecture 

#### Control Objects
To enable user control, the program uses custom Control Objects:
- ```Parameters``` are numerical data in a range (>2) of minimum and maximum values. Think of these like faders or potentiometers. Hue (0-179) and Saturation (0-254) are examples.
- ```Toggles``` are boolean. Think of these as toggle buttons or flags. Example: Enable Polar Transform, 

These are primarily used by the effects classes, but are also used in the mixer and oscillator bank. Essentially any single value that a user can alter should be a Control Object.

#### Control Structures
The ```Param``` and ```Toggle``` Control Objects are managed by ```ParamsTable``` and ```ButtonsTable```  Control Structures respectively.

These structures must be passed to each class that wishes to permit user control, oscillator linking, gui sliders/buttons, effect sequencing, etc.

#### Effects Classes and Effects Manager
For effect modularity and gui simplicity, each related set of effects are placed into an Effects Class.
Each effect class is derived from the ```EffectBase``` parent class.
Each effect class should take the ```ParamsTable``` and ```ButtonsTable``` as arugments, add its own params, and implement its own ```create_gui_panel``` method. Until this is automated, the ```create_gui_panel``` must be manually called in ```gui.py/create_interface()```***.

All effects classes are stored, initialized and managed by the ```EffectsManager``` facade class. This simplifies dependencies, arguments, and effect sequencing. Feedback is currently ommitted from effect sequencing until further experimentation. 

## Background and Inspiration

My journey into video art was inspired by [this video](https://www.youtube.com/watch?v=D3eHKI0nvKA), which was graciously provided by the algorithm. In describing his kinetic video feedback synthesizer masterpeice, Dave Blair explains why his machine must use an expensive field-monitor commonly used in movie production, as it offers analog knobs to manipulate image properties. While this requirement makes sense for his application, I was inspired to explore a purely code-based solution, with the original intention to solder up an HID-capable MCU + encoders and interface it to a basic openCV program.

I have since pivoted to using off-the-shelf MIDI controllers to emulate effects normally acheived through analog video synthesis modules and mixers (feedback, oscillators, sync modulation), as well as early digital animation techniques (perlin noise, fractal noise, [metaballs](https://steve.hollasch.net/cgindex/misc/metaballs.html), plasma, moire, etc.), op art and stuff. 

## Optional Off-The-Shelf Hardware

These are cheap, readily available components that help with converting and capturing various common formats. 

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

These tools require a combination of off-the-shelf components, discrete component, DIY assembly, and enclosure design

- GBS Video Feedback Synth (instructions by [LofiFuture]())
    - more bend points for 74HC identified by [Gleix]()
- 2-in-1-out passive composite switcher
- Klompf Dirty Video Mixer
- VC9900 Glitch Box (bend points from the GBS Feedback Synth apply here) w/ [GBSControl]()

## Optional Custom PCBs

These are custom PCBs that I have built and utilize. They may be available fully assembled in online shops, but they can also be manufactured and assembled DIY style.

- CHA/V by [Jonas Bers](https://jonasbers.com/chav/)
- Archer Experimenter by [lightmusicvisuals](https://github.com/lightmusicvisuals/archer_experimenter)
- PixelSlasher by [CTXz](https://github.com/CTXz/Video-Glitch-Array/tree/master/PixelSlasher)
- recurBOY by [cyberboy666](https://github.com/cyberboy666/recurBOY)
- _rupture_ by [cyberboy666](https://github.com/cyberboy666/_rupture_)
- sync_ope by [cyberboy666](https://github.com/cyberboy666/sync_ope)

## Control
The app can be controlled through a (crappy) GUI or midi controller. The program is currently configured to use a cheap AKAI MIDI Mix or MVAVE SMC Mixer, but there are provisions to configure a new controller with relative ease. To implement a new controller, see the required functions you must expose in the example ```SMC_Mixer()``` or ```MidiMix()``` classes in ```midi.py```. Most work involves the mapping.

Note that the order that the controllers are initialized is extremely important. In ```main.py```, the order that you initialize your controllers must reflect the order in which the controllers are plugged into your PC. For example, if the ```SMC_Mixer()``` is plugged in first, but initialized second, controller identification will fail.


## My Current Configuration
* TODO: include VGA capture device in sources
* TODO: include CHA/V, link to VGA capture source
* TODO: add posterize, solarize, to effects
* note that the Unmixed HDMI Display is not the fully mixed video. It is essentially a placeholder to indicate that there is another HDMI output available.

![Video Synthesizer Diagram](documentation/diagram.svg)

