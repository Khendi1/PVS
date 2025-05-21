class Timeline:
    def __init__(self, sliders):
        self.labels = [s.labels for s in sliders]
        self.current_state = [s.value for s in sliders]
        self.prev_state = self.current_state

    def update(self, sliders=None, buttons=None):
        if sliders is not None:
            self.labels = [s.labels for s in sliders]
            self
            self.current_state = [s.value for s in sliders]
        self.prev_state = self.current_state
        self.current_state = [s.value for s in sliders]
