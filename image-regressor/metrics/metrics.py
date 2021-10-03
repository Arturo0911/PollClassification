
class RunningMetrics:
    def __init__(self):
        self.N = 0
        self.S = 0

    def update(self, val: float, size: int):
        self.N += val
        self.S += size

    def __call__(self):
        return self.N / float(self.S)