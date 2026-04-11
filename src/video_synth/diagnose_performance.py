#!/usr/bin/env python3
"""
Performance diagnostic script for video synthesizer.
Run this to identify specific bottlenecks causing stuttering.
"""

import sys
import time
import cProfile
import pstats
from io import StringIO

# Add video_synth to path
sys.path.insert(0, 'video_synth')

NUM_FRAMES_TO_PROFILE = 120 # 120 frames at 30 FPS = 4 seconds of profiling
NUM_STATS_TO_SHOW = 50 # Adjust this threshold based on your performance goals

def profile_frame_processing():
    """Profile a single frame processing cycle"""
    from mixer import Mixer
    from effects_manager import EffectManager
    from common import Groups, MixerSource

    print("Initializing components...")

    # Initialize effect managers
    src_1_effects = EffectManager(Groups.SRC_1_EFFECTS, 640, 480)
    src_2_effects = EffectManager(Groups.SRC_2_EFFECTS, 640, 480)
    post_effects = EffectManager(Groups.POST_EFFECTS, 640, 480)
    effects = (src_1_effects, src_2_effects, post_effects)

    mixer = Mixer(effects, num_devices=0, width=640, height=480)

    print("Warming up (first frame often slower)...")
    wet_frame = dry_frame = mixer.get_frame()
    if dry_frame is None:
        print("ERROR: Could not get initial frame")
        return

    prev_frame = dry_frame.copy()

    # Process a few frames to warm up
    for i in range(5):
        dry_frame = mixer.get_frame()
        for effect_manager in effects:
            effect_manager.oscs.update()
        prev_frame, wet_frame = effects[MixerSource.POST.value].get_frames(
            dry_frame, wet_frame, prev_frame, i
        )

    print(f"\nProfiling {NUM_FRAMES_TO_PROFILE} frames...")

    # Profile 30 frames
    profiler = cProfile.Profile()
    profiler.enable()

    start = time.perf_counter()
    for i in range(NUM_FRAMES_TO_PROFILE): # Process 30 frames to get a good sample size
        dry_frame = mixer.get_frame()
        for effect_manager in effects:
            effect_manager.oscs.update()
        prev_frame, wet_frame = effects[MixerSource.POST.value].get_frames(
            dry_frame, wet_frame, prev_frame, i
        )

    elapsed = time.perf_counter() - start
    profiler.disable()

    # Print results
    fps = NUM_FRAMES_TO_PROFILE / elapsed
    avg_frame_time = (elapsed / NUM_FRAMES_TO_PROFILE) * 1000

    print(f"\n{'='*60}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Total time for {NUM_FRAMES_TO_PROFILE} frames: {elapsed:.3f}s")
    print(f"Average FPS: {fps:.1f}")
    print(f"Average frame time: {avg_frame_time:.1f}ms")
    print(f"Target avg frame time for 30 FPS: 33.3ms")

    if avg_frame_time > 33.3:
        print(f"⚠️  BOTTLENECK DETECTED: {avg_frame_time:.1f}ms > 33.3ms")
        print(f"   Frames are taking {(avg_frame_time/33.3 - 1)*100:.1f}% longer than target")
    else:
        print(f"✓  Frame time is good: {avg_frame_time:.1f}ms < 33.3ms")

    # Detailed profiling stats
    print(f"\n{'='*60}")
    print(f"TOP {NUM_STATS_TO_SHOW} SLOWEST FUNCTIONS")
    print(f"{'='*60}")

    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(NUM_STATS_TO_SHOW)

    # Filter and print relevant lines
    for line in s.getvalue().split('\n'):
        if 'video_synth' in line or 'ncalls' in line or '----' in line:
            print(line)

    print(f"\n{'='*60}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*60}")

    if avg_frame_time > 50:
        print("❌ SEVERE STUTTERING EXPECTED")
        print("   - Look for functions taking >5ms in the profile above")
        print("   - Check if heavy effects are enabled (sync, frame buffer)")
    elif avg_frame_time > 40:
        print("⚠️  MODERATE STUTTERING LIKELY")
        print("   - Consider disabling some effects")
        print("   - Check for parameter access contention")
    elif avg_frame_time > 33.3:
        print("⚠️  MINOR FRAME DROPS POSSIBLE")
        print("   - Should be mostly smooth but occasional stutters")
    else:
        print("✓  Performance looks good!")

    # Cleanup
    mixer.cleanup()
    print("\nDone!")


if __name__ == "__main__":
    try:
        profile_frame_processing()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
