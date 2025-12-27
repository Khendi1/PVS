class ScalingHelper:
    @staticmethod
    def float_to_int(f: float, scale_factor: int = 1000) -> int:
        """Converts a float to an integer for a slider, maintaining precision."""
        return int(f * scale_factor)

    @staticmethod
    def int_to_float(i: int, scale_factor: int = 1000) -> float:
        """Converts an integer from a slider back to a float."""
        return i / scale_factor
