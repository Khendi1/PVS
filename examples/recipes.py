"""
Parameter recipes for creating melty, feedback-driven visual effects.
Each recipe sets a combination of parameters via the API to produce
a specific aesthetic. Run the video synth with --api first.

Usage:
    python examples/recipes.py [recipe_name]
    python examples/recipes.py --list
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'video_synth'))

import requests
import time

API_BASE = 'http://127.0.0.1:8000'


# ── Recipe definitions ──────────────────────────────────────────────

RECIPES = {
    "wax_melt": {
        "description": "Slow luminance-driven displacement. Bright areas push outward like melting wax.",
        "params": {
            "alpha": 0.75,
            "warp_type": 7,           # DISPLACEMENT
            "disp_strength": 8.0,
            "disp_decay": 0.92,
            "disp_blur": 5,
            "fb_paint_zoom": 1.002,
            "temporal_filter": 0.3,
        }
    },

    "candle_flame": {
        "description": "Heat-rise convection. Patterns dissolve upward like candle smoke.",
        "params": {
            "alpha": 0.8,
            "warp_type": 8,           # CONVECTION
            "conv_rise_speed": 3.0,
            "conv_diffusion": 0.6,
            "conv_turbulence": 0.4,
            "conv_decay": 0.93,
            "temporal_filter": 0.2,
        }
    },

    "organic_breathe": {
        "description": "Reaction-diffusion displacement creates organic cell-boundary warps.",
        "params": {
            "alpha": 0.7,
            "warp_type": 9,           # RD_WARP
            "rd_warp_strength": 12.0,
            "rd_warp_feed": 0.055,
            "rd_warp_kill": 0.062,
            "rd_warp_speed": 1.0,
            "temporal_filter": 0.15,
        }
    },

    "feedback_chaos": {
        "description": "Classic feedback warp with paint drift for spiraling fractal chaos.",
        "params": {
            "alpha": 0.85,
            "warp_type": 6,           # FEEDBACK
            "fb_warp_decay": 0.96,
            "fb_warp_strength": 15.0,
            "fb_warp_freq": 5.0,
            "fb_paint_drift_x": 1.5,
            "fb_paint_drift_y": -0.5,
            "fb_paint_rotation": 0.3,
            "fb_paint_zoom": 1.003,
        }
    },

    "lava_lamp": {
        "description": "Slow convection with high diffusion and color cycling.",
        "params": {
            "alpha": 0.9,
            "warp_type": 8,           # CONVECTION
            "conv_rise_speed": 1.5,
            "conv_diffusion": 0.8,
            "conv_turbulence": 0.2,
            "conv_decay": 0.97,
            "color_cycle_speed": 0.3,
            "color_cycle_bands": 4,
            "hue_shift": 50,
        }
    },

    "acid_dream": {
        "description": "Displacement feedback + erosion + color cycling for psychedelic visuals.",
        "params": {
            "alpha": 0.8,
            "warp_type": 7,           # DISPLACEMENT
            "disp_strength": 12.0,
            "disp_decay": 0.88,
            "disp_blur": 3,
            "erosion_strength": 0.4,
            "erosion_speed": 0.5,
            "color_cycle_speed": 0.5,
            "color_cycle_bands": 6,
            "fb_paint_rotation": 0.5,
        }
    },

    "slow_dissolve": {
        "description": "Gentle displacement with heavy temporal filtering for dreamlike dissolves.",
        "params": {
            "alpha": 0.6,
            "warp_type": 7,           # DISPLACEMENT
            "disp_strength": 4.0,
            "disp_decay": 0.95,
            "disp_blur": 7,
            "temporal_filter": 0.6,
            "fb_paint_zoom": 1.001,
        }
    },

    "reset": {
        "description": "Reset all recipe-related parameters to defaults.",
        "params": {
            "alpha": 0.0,
            "warp_type": 0,
            "temporal_filter": 0.0,
            "fb_paint_drift_x": 0.0,
            "fb_paint_drift_y": 0.0,
            "fb_paint_rotation": 0.0,
            "fb_paint_zoom": 1.0,
            "erosion_strength": 0.0,
            "color_cycle_speed": 0.0,
        }
    },
}


# ── Helpers ─────────────────────────────────────────────────────────

def check_connection():
    try:
        return requests.get(f'{API_BASE}/', timeout=2).status_code == 200
    except requests.ConnectionError:
        return False


def apply_recipe(name):
    recipe = RECIPES.get(name)
    if not recipe:
        print(f"Unknown recipe: {name}")
        print(f"Available: {', '.join(RECIPES.keys())}")
        return False

    print(f"Applying recipe: {name}")
    print(f"  {recipe['description']}")

    for param, value in recipe['params'].items():
        try:
            r = requests.put(f'{API_BASE}/params/{param}', json={'value': value}, timeout=2)
            if r.status_code != 200:
                print(f"  WARNING: Failed to set {param}={value} (status {r.status_code})")
        except requests.ConnectionError:
            print(f"  ERROR: Lost connection to API")
            return False

    print(f"  Applied {len(recipe['params'])} parameters.")
    return True


def list_recipes():
    print("Available recipes:")
    print("-" * 60)
    for name, recipe in RECIPES.items():
        print(f"  {name:20s} {recipe['description']}")
    print("-" * 60)


# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not check_connection():
        print("ERROR: Cannot connect to video synth API at", API_BASE)
        print("Start the synth with: python video_synth --api")
        sys.exit(1)

    if len(sys.argv) < 2 or sys.argv[1] == '--list':
        list_recipes()
        sys.exit(0)

    recipe_name = sys.argv[1]
    apply_recipe(recipe_name)
