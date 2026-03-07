import cv2
import numpy as np
import logging
import time

from effects.base import EffectBase
from effects.enums import HSV
from animations.enums import Colormap, COLORMAP_OPTIONS
from common import Widget

log = logging.getLogger(__name__)

class Color(EffectBase):

    def __init__(self, params, group):
        subgroup = self.__class__.__name__
        self.params = params
        self.group = group

        self.hue_shift = params.add("hue_shift",
                                    min=0, max=180, default=0,
                                    subgroup=subgroup, group=group)
        self.sat_shift = params.add("sat_shift",
                                    min=0, max=255, default=0,
                                    subgroup=subgroup, group=group)
        self.val_shift = params.add("val_shift",
                                    min=0, max=255, default=0,
                                    subgroup=subgroup, group=group)

        self.levels_per_channel = params.add("posterize_levels",
                                             min=0, max=100, default=0.0,
                                             subgroup=subgroup, group=group)
        self.num_hues = params.add("num_hues",
                                   min=2, max=10, default=8,
                                   subgroup=subgroup, group=group)

        self.val_threshold = params.add("val_threshold",
                                        min=0, max=255, default=0,
                                        subgroup=subgroup, group=group)
        self.val_hue_shift = params.add("val_hue_shift",
                                        min=0, max=255, default=0,
                                        subgroup=subgroup, group=group)

        self.solarize_threshold = params.add("solarize_threshold",
                                             min=0, max=100, default=0.0,
                                             subgroup=subgroup, group=group)
        self.hue_invert_angle = params.add("hue_invert_angle",
                                           min=0, max=360, default=0,
                                           subgroup=subgroup, group=group)
        self.hue_invert_strength = params.add("hue_invert_strength",
                                              min=0.0, max=1.0, default=0.0,
                                              subgroup=subgroup, group=group)

        self.contrast = params.add("contrast",
                                   min=0.5, max=3.0, default=1.0,
                                   subgroup=subgroup, group=group)
        self.brightness = params.add("brightness",
                                     min=0, max=100, default=0,
                                     subgroup=subgroup, group=group)
        self.gamma = params.add("gamma",
                                min=0.1, max=3.0, default=1.0,
                                subgroup=subgroup, group=group)
        self.highlight_compression = params.add("highlight_compression",
                                                min=0.0, max=1.0, default=0.0,
                                                subgroup=subgroup, group=group)

        # Color cycling - rotates a palette ramp over the brightness of the image
        self.color_cycle_speed = params.add("color_cycle_speed",
                                             min=0.0, max=5.0, default=0.0,
                                             subgroup=subgroup, group=group)
        self.color_cycle_bands = params.add("color_cycle_bands",
                                             min=1, max=8, default=3,
                                             subgroup=subgroup, group=group)

        # Channel Mixer - cross-mix RGB channels
        self.ch_mix_rr = params.add("ch_mix_rr", min=0.0, max=2.0, default=1.0,
                                     subgroup=subgroup, group=group)
        self.ch_mix_rg = params.add("ch_mix_rg", min=0.0, max=2.0, default=0.0,
                                     subgroup=subgroup, group=group)
        self.ch_mix_rb = params.add("ch_mix_rb", min=0.0, max=2.0, default=0.0,
                                     subgroup=subgroup, group=group)
        self.ch_mix_gr = params.add("ch_mix_gr", min=0.0, max=2.0, default=0.0,
                                     subgroup=subgroup, group=group)
        self.ch_mix_gg = params.add("ch_mix_gg", min=0.0, max=2.0, default=1.0,
                                     subgroup=subgroup, group=group)
        self.ch_mix_gb = params.add("ch_mix_gb", min=0.0, max=2.0, default=0.0,
                                     subgroup=subgroup, group=group)
        self.ch_mix_br = params.add("ch_mix_br", min=0.0, max=2.0, default=0.0,
                                     subgroup=subgroup, group=group)
        self.ch_mix_bg = params.add("ch_mix_bg", min=0.0, max=2.0, default=0.0,
                                     subgroup=subgroup, group=group)
        self.ch_mix_bb = params.add("ch_mix_bb", min=0.0, max=2.0, default=1.0,
                                     subgroup=subgroup, group=group)

        # Color Bitcrush - reduce bit depth per channel
        self.color_bitcrush = params.add("color_bitcrush",
                                          min=1, max=8, default=8,
                                          subgroup=subgroup, group=group)

        # Hue Scatter - randomize hue per-pixel
        self.hue_scatter = params.add("hue_scatter",
                                       min=0.0, max=1.0, default=0.0,
                                       subgroup=subgroup, group=group)

        # Duotone - map luminance to two-color gradient
        self.duotone_strength = params.add("duotone_strength",
                                            min=0.0, max=1.0, default=0.0,
                                            subgroup=subgroup, group=group)
        self.duotone_hue_lo = params.add("duotone_hue_lo",
                                          min=0, max=180, default=120,
                                          subgroup=subgroup, group=group)
        self.duotone_hue_hi = params.add("duotone_hue_hi",
                                          min=0, max=180, default=10,
                                          subgroup=subgroup, group=group)

        # Channel Isolation - mute individual R/G/B channels
        self.ch_r = params.add("ch_r", min=0.0, max=1.0, default=1.0,
                                subgroup=subgroup, group=group)
        self.ch_g = params.add("ch_g", min=0.0, max=1.0, default=1.0,
                                subgroup=subgroup, group=group)
        self.ch_b = params.add("ch_b", min=0.0, max=1.0, default=1.0,
                                subgroup=subgroup, group=group)

        # Chromatic Aberration - offset R/G/B channels spatially
        self.chroma_ab_x = params.add("chroma_ab_x",
                                       min=0, max=30, default=0,
                                       subgroup=subgroup, group=group)
        self.chroma_ab_y = params.add("chroma_ab_y",
                                       min=0, max=30, default=0,
                                       subgroup=subgroup, group=group)

        # Color Temperature - warm/cool shift
        self.color_temp = params.add("color_temp",
                                      min=-1.0, max=1.0, default=0.0,
                                      subgroup=subgroup, group=group)

        # Saturation Curve - non-linear saturation boost/crush
        self.sat_curve_shadows = params.add("sat_curve_shadows",
                                             min=0.0, max=3.0, default=1.0,
                                             subgroup=subgroup, group=group)
        self.sat_curve_mids = params.add("sat_curve_mids",
                                          min=0.0, max=3.0, default=1.0,
                                          subgroup=subgroup, group=group)
        self.sat_curve_highlights = params.add("sat_curve_highlights",
                                                min=0.0, max=3.0, default=1.0,
                                                subgroup=subgroup, group=group)

        # False Color - apply colormap to luminance
        self.false_color_strength = params.add("false_color_strength",
                                                min=0.0, max=1.0, default=0.0,
                                                subgroup=subgroup, group=group)
        self.false_color_map = params.add("false_color_map",
                                           min=0, max=len(COLORMAP_OPTIONS)-1,
                                           default=int(Colormap.INFERNO),
                                           subgroup=subgroup, group=group,
                                           type=Widget.DROPDOWN, options=Colormap)

        # Invert - partial or full color inversion
        self.invert_strength = params.add("invert_strength",
                                           min=0.0, max=1.0, default=0.0,
                                           subgroup=subgroup, group=group)

    def _shift_hue(self, hue: int):
        """
        Shifts the hue of an image by a specified amount, wrapping aroung in necessary.
        """
        return (hue + self.hue_shift.value) % 180

    def _shift_sat(self, sat: int):
        """
        Shifts the saturation of an image by a specified amount, clamping to [0, 255].
        """
        return np.clip(sat + self.sat_shift.value, 0, 255)

    def _shift_val(self, val: int):
        """
        Shifts the value of an image by a specified amount, clamping to [0, 255].
        """
        return np.clip(val + self.val_shift.value, 0, 255)

    def _shift_hsv(self, hsv: list):
        """
        Shifts the hue, saturation, and value of an image by specified amounts.
        """
        return [
            self._shift_hue(hsv[HSV.H]),
            self._shift_sat(hsv[HSV.S]),
            self._shift_val(hsv[HSV.V]),
        ]

    def _val_threshold_hue_shift(self, hsv: list):
        """
        Shifts the hue of pixels in an image that have a saturation value
        greater than the given threshold.

        Args:
            image (numpy.ndarray): The input BGR image.
            val_threshold (int): The minimum saturation value for hue shifting.
            hue_shift (int): The amount to shift the hue.

        Returns:
            numpy.ndarray: The output image with shifted hues.
        """

        # Create a mask for pixels with saturation above the threshold.
        mask = hsv[HSV.V.value] > self.val_threshold.value

        # Shift and wrap around the hue values for the masked pixels
        hsv[HSV.H.value][mask] = (
            hsv[HSV.H.value][mask] + self.val_hue_shift.value
        ) % 180

        return hsv

    def modify_hsv(self, image: np.ndarray):
        if (self.hue_shift.value == 0 and
            self.sat_shift.value == 0 and
            self.val_shift.value == 0 and
            self.val_threshold.value == 0):
            return image
        
        is_float = image.dtype == np.float32
        if is_float:
            image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
        else:
            image_uint8 = image
            
        hsv_image = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        hsv = [h, s, v] # Create a list of channels for _shift_hsv
        
        # Apply the hue, saturation, and value shifts to the image.
        hsv = self._shift_hsv(hsv)
        hsv = self._val_threshold_hue_shift(hsv)

        # Merge the modified channels and convert back to BGR color space.
        result_uint8 = cv2.cvtColor(
            cv2.merge((hsv[HSV.H.value], hsv[HSV.S.value], hsv[HSV.V.value])),
            cv2.COLOR_HSV2BGR,
        )

        if is_float:
            return result_uint8.astype(np.float32)
        else:
            return result_uint8

    def limit_hues_kmeans(self, frame: np.ndarray):

        if self.num_hues.value <= self.num_hues.max:
            return frame

        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Reshape the image to a 2D array of pixels (N x 3) for K-Means
        # We'll focus on the Hue channel, so we can reshape to (N x 1) if we only want to cluster hue
        # For a more general color quantization (hue, saturation, value), keep it N x 3

        # Option 2: Quantize all HSV channels (more common for general color reduction)
        # This will result in a limited set of overall colors, which inherently limits hues.
        data = frame.reshape((-1, 3)).astype(np.float32)

        # Define criteria for K-Means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # Apply K-Means clustering
        # attempts: Number of times the algorithm is executed with different initial labellings.
        #           The best result of all attempts is returned.
        # flags: Specifies how initial centers are chosen. KMEANS_PP_CENTERS is a good choice.
        compactness, labels, centers = cv2.kmeans(
            data, self.num_hues.value, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        # Convert back to uint8 and reshape to original image dimensions
        centers = np.uint8(centers)
        quantized_data = centers[labels.flatten()]
        quantized_image = quantized_data.reshape(hsv_image.shape)

        # Convert back to BGR for display
        output_image = cv2.cvtColor(quantized_image, cv2.COLOR_HSV2BGR)
        return output_image

    def posterize(self, frame: np.ndarray):
        """
        Applies a posterization effect to an image.

        Args:
            image_path (str): The path to the input image.
            levels_per_channel (int): The number of distinct intensity levels
                                    to use for each color channel (e.g., 8, 16, 32).
                                    Must be between 2 and 255.
        Returns:
            numpy.ndarray: The posterized image.
        """
        if self.levels_per_channel.value <= self.levels_per_channel.min:
            return frame

        # Ensure the image is in 8-bit format (0-255)
        if frame.dtype != np.uint8:
            # log.debug("Warning: Image not in uint8 format. Converting...")
            frame = cv2.convertScaleAbs(frame)

        # Calculate the step size for quantization, round
        step = 256 // self.levels_per_channel.value
        half_step = step // 2

        # Method 2: Quantization with rounding (often looks better)
        posterized_frame = ((frame + half_step) // step) * step

        # Ensure values stay within 0-255.
        posterized_frame = np.clip(posterized_frame, 0, 255).astype(np.uint8)

        return posterized_frame

    def solarize_image(self, frame: np.ndarray):
        """
        Applies a solarize effect to an image.

        Args:
            image_path (str): The path to the input image.
            threshold (int): The intensity threshold (0-255). Pixels above this
                            value will be inverted. Default is 128 (mid-range).

        Returns:
            numpy.ndarray: The solarized image.
        """
        if self.solarize_threshold.value == 0:
            return frame

        # Ensure the image is in 8-bit format (0-255)
        if frame.dtype != np.uint8:
            # log.debug("Warning: Image not in uint8 format. Converting...")
            frame = cv2.convertScaleAbs(frame)  # Converts to uint8, scales if needed

        frame = np.where(frame > self.solarize_threshold.value, 255 - frame, frame)

        return frame.astype(np.uint8)

    def adjust_brightness_contrast(self, image):
        """
        Adjusts the brightness and contrast of an image using float precision.
        """
        if self.contrast.value == 1.0 and self.brightness.value == 0:
            return image

        # Ensure we are working with floats for this calculation
        if image.dtype != np.float32:
            image = image.astype(np.float32)
            
        # Perform the calculation in float32.
        # Clipping will be handled once at the end of the get_frames function.
        # return image * self.contrast.value + self.brightness.value
        return cv2.convertScaleAbs(
                image, alpha=self.contrast.value, beta=self.brightness.value
            ).astype(np.float32)

    def polarize_frame_hsv(self, frame: np.ndarray):
        """
        Polarizes a frame by rotating hue in HSV color space.  This often gives
        a more visually interesting effect than rotating in BGR.

        Args:
            frame (numpy.ndarray): The input frame as a NumPy array (H, W, 3) in BGR format.
            angle (float): The polarization angle in degrees.
            strength (float): The strength of the polarization effect (0 to 1).

        Returns:
            numpy.ndarray: The polarized frame as a NumPy array (H, W, 3) in BGR format.
        """
        if self.hue_invert_strength.value <= self.hue_invert_strength.min:
            return frame

        # Convert to HSV color space and extract hsv channel
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue_channel = hsv_frame[:, :, 0]

        # Convert angle to OpenCV hue units (0-180)
        hue_shift = (self.hue_invert_angle.value / 360.0) * 180

        # Apply the hue shift with strength
        shifted_hue = (
            hue_channel + hue_shift * self.hue_invert_strength.value
        ) % 180  # Wrap around
        hsv_frame[:, :, 0] = shifted_hue

        # Convert back to BGR
        polarized_frame = cv2.cvtColor(hsv_frame.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return polarized_frame

    def adjust_gamma(self, image: np.ndarray):
        """
        Applies gamma correction to the image.
        """
        gamma = self.gamma.value
        if gamma == 1.0:
            return image

        # Ensure we are working with floats
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Normalize to [0, 1], apply gamma, then scale back to [0, 255]
        # The 1e-6 avoids division by zero for black pixels.
        return np.power(image / 255.0, gamma) * 255.0


    def tonemap(self, image: np.ndarray):
        """
        Applies Reinhard tonemapping to compress highlights and prevent clipping.
        """
        strength = self.highlight_compression.value
        if strength == 0.0:
            return image  # PERFORMANCE: Return unchanged when disabled

        # Map the [0,1] strength slider to the intensity parameter of the tonemapper.
        # Negative intensity values decrease brightness.
        tonemap_intensity = (strength * -8.0)

        tonemapper = cv2.createTonemapReinhard(gamma=2.2, intensity=tonemap_intensity, light_adapt=0.0, color_adapt=0.0)
        
        # The tonemapper expects a float32 BGR image, which is what we have.
        # It processes the image and returns it mapped to the [0, 1] range.
        ldr_image = tonemapper.process(image)
        
        # Scale the result back up to the [0, 255] range.
        return ldr_image * 255.0

    def color_cycle(self, frame: np.ndarray):
        """
        Rotates a color palette over the brightness of the image.
        Maps luminance to a cycling hue ramp — static textures come alive
        as the palette rotates through them.
        """
        speed = self.color_cycle_speed.value
        if speed == 0.0:
            return frame

        bands = int(self.color_cycle_bands.value)
        offset = time.time() * speed * 30.0  # continuous rotation

        # Convert to grayscale luminance
        if frame.dtype != np.uint8:
            work = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            work = frame

        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

        # Map luminance through cycling hue ramp
        hue = ((gray.astype(np.float32) * bands + offset) % 180).astype(np.uint8)
        sat = np.full_like(gray, 200, dtype=np.uint8)
        val = gray  # preserve original brightness

        hsv = cv2.merge([hue, sat, val])
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if frame.dtype == np.float32:
            return result.astype(np.float32)
        return result

    def channel_mix(self, frame: np.ndarray):
        """Cross-mix RGB channels via a 3x3 matrix. Identity = no change."""
        rr = self.ch_mix_rr.value
        rg = self.ch_mix_rg.value
        rb = self.ch_mix_rb.value
        gr = self.ch_mix_gr.value
        gg = self.ch_mix_gg.value
        gb = self.ch_mix_gb.value
        br = self.ch_mix_br.value
        bg = self.ch_mix_bg.value
        bb = self.ch_mix_bb.value

        # Early exit if identity matrix
        if (rr == 1.0 and rg == 0.0 and rb == 0.0 and
            gr == 0.0 and gg == 1.0 and gb == 0.0 and
            br == 0.0 and bg == 0.0 and bb == 1.0):
            return frame

        if frame.dtype != np.float32:
            work = frame.astype(np.float32)
        else:
            work = frame.copy()

        b, g, r = work[:, :, 0], work[:, :, 1], work[:, :, 2]
        new_b = bb * b + bg * g + br * r
        new_g = gb * b + gg * g + gr * r
        new_r = rb * b + rg * g + rr * r

        result = np.stack([new_b, new_g, new_r], axis=2)
        return np.clip(result, 0, 255)

    def color_bitcrush(self, frame: np.ndarray):
        """Reduce color bit depth per channel for hard RGB banding."""
        bits = int(self.color_bitcrush.value)
        if bits >= 8:
            return frame

        if frame.dtype != np.uint8:
            work = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            work = frame

        # Shift right then left to zero out lower bits
        shift = 8 - bits
        result = (work >> shift) << shift

        if frame.dtype == np.float32:
            return result.astype(np.float32)
        return result

    def hue_scatter(self, frame: np.ndarray):
        """Randomize hue per-pixel proportional to strength. Creates chromatic shimmer."""
        strength = self.hue_scatter.value
        if strength == 0.0:
            return frame

        if frame.dtype != np.uint8:
            work = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            work = frame.copy()

        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].astype(np.int16)

        noise = np.random.randint(-90, 91, h.shape, dtype=np.int16)
        h = (h + (noise * strength).astype(np.int16)) % 180
        hsv[:, :, 0] = h.astype(np.uint8)

        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if frame.dtype == np.float32:
            return result.astype(np.float32)
        return result

    def duotone(self, frame: np.ndarray):
        """Map luminance to a two-color gradient defined by lo/hi hue."""
        strength = self.duotone_strength.value
        if strength == 0.0:
            return frame

        hue_lo = int(self.duotone_hue_lo.value)
        hue_hi = int(self.duotone_hue_hi.value)

        if frame.dtype != np.uint8:
            work = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            work = frame

        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        t = gray.astype(np.float32) / 255.0  # 0=shadow, 1=highlight

        # Interpolate hue (handle wrap-around)
        if abs(hue_hi - hue_lo) > 90:
            # Wrap the shorter way around
            if hue_lo > hue_hi:
                hue = (hue_lo + t * (hue_hi + 180 - hue_lo)) % 180
            else:
                hue = (hue_hi + (1 - t) * (hue_lo + 180 - hue_hi)) % 180
        else:
            hue = hue_lo + t * (hue_hi - hue_lo)

        hue = hue.astype(np.uint8)
        sat = np.full_like(gray, 220, dtype=np.uint8)

        hsv = cv2.merge([hue, sat, gray])
        duo = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Blend with original
        if strength < 1.0:
            result = cv2.addWeighted(work, 1.0 - strength, duo, strength, 0)
        else:
            result = duo

        if frame.dtype == np.float32:
            return result.astype(np.float32)
        return result

    def channel_isolate(self, frame: np.ndarray):
        """Scale individual R/G/B channels. Set to 0 to mute, 1 = unchanged."""
        r_scale = self.ch_r.value
        g_scale = self.ch_g.value
        b_scale = self.ch_b.value

        if r_scale == 1.0 and g_scale == 1.0 and b_scale == 1.0:
            return frame

        if frame.dtype != np.float32:
            work = frame.astype(np.float32)
        else:
            work = frame.copy()

        work[:, :, 0] *= b_scale
        work[:, :, 1] *= g_scale
        work[:, :, 2] *= r_scale

        return np.clip(work, 0, 255)

    def chromatic_aberration(self, frame: np.ndarray):
        """Offset R/G/B channels spatially for lens distortion look."""
        offset_x = int(self.chroma_ab_x.value)
        offset_y = int(self.chroma_ab_y.value)

        if offset_x == 0 and offset_y == 0:
            return frame

        if frame.dtype != np.uint8:
            work = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            work = frame.copy()

        h, w = work.shape[:2]
        b, g, r = cv2.split(work)

        # Shift R channel right/down, B channel left/up, G stays centered
        M_r = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        M_b = np.float32([[1, 0, -offset_x], [0, 1, -offset_y]])

        r_shifted = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REFLECT)
        b_shifted = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REFLECT)

        result = cv2.merge([b_shifted, g, r_shifted])

        if frame.dtype == np.float32:
            return result.astype(np.float32)
        return result

    def color_temperature(self, frame: np.ndarray):
        """Warm/cool white balance shift. Positive = warm (orange), negative = cool (blue)."""
        temp = self.color_temp.value
        if temp == 0.0:
            return frame

        if frame.dtype != np.float32:
            work = frame.astype(np.float32)
        else:
            work = frame.copy()

        # Warm: boost red, reduce blue. Cool: boost blue, reduce red.
        work[:, :, 2] += temp * 30.0   # R channel
        work[:, :, 1] += temp * 10.0   # G channel (slight warm tint)
        work[:, :, 0] -= temp * 30.0   # B channel

        return np.clip(work, 0, 255)

    def saturation_curve(self, frame: np.ndarray):
        """Non-linear saturation: independently control shadows, mids, highlights."""
        s_shadows = self.sat_curve_shadows.value
        s_mids = self.sat_curve_mids.value
        s_highs = self.sat_curve_highlights.value

        if s_shadows == 1.0 and s_mids == 1.0 and s_highs == 1.0:
            return frame

        if frame.dtype != np.uint8:
            work = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            work = frame.copy()

        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.float32) / 255.0
        s = hsv[:, :, 1].astype(np.float32)

        # Smooth weight masks for shadows (<0.33), mids (0.33-0.66), highlights (>0.66)
        w_shadow = np.clip(1.0 - v * 3.0, 0, 1)
        w_high = np.clip(v * 3.0 - 2.0, 0, 1)
        w_mid = 1.0 - w_shadow - w_high

        multiplier = w_shadow * s_shadows + w_mid * s_mids + w_high * s_highs
        s = np.clip(s * multiplier, 0, 255).astype(np.uint8)
        hsv[:, :, 1] = s

        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        if frame.dtype == np.float32:
            return result.astype(np.float32)
        return result

    def false_color(self, frame: np.ndarray):
        """Apply a colormap to luminance for thermal/scientific visualization look."""
        strength = self.false_color_strength.value
        if strength == 0.0:
            return frame

        if frame.dtype != np.uint8:
            work = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            work = frame

        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        cmap_idx = int(self.false_color_map.value)
        colored = cv2.applyColorMap(gray, COLORMAP_OPTIONS[cmap_idx])

        if strength < 1.0:
            result = cv2.addWeighted(work, 1.0 - strength, colored, strength, 0)
        else:
            result = colored

        if frame.dtype == np.float32:
            return result.astype(np.float32)
        return result

    def invert(self, frame: np.ndarray):
        """Partial or full color inversion with blend control."""
        strength = self.invert_strength.value
        if strength == 0.0:
            return frame

        if frame.dtype != np.float32:
            work = frame.astype(np.float32)
        else:
            work = frame

        inverted = 255.0 - work

        if strength < 1.0:
            result = work * (1.0 - strength) + inverted * strength
        else:
            result = inverted

        return np.clip(result, 0, 255)
