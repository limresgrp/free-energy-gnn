

class EarlyStopping:
    """EarlyStopping handler can be used to stop the training if no improvement after a given number of epochs."""

    def __init__(self, patience: int):
        self.patience = patience
        self.min_delta = 0.01  # Minimum gain to be considered an improvement
        self.losses = []
        self.best_score = None
        self.counter = 0

    def check(self, loss):
        """Return bool whether we should stop the training or not"""
        self.losses.append(loss)

        if self.counter > self.patience:
            return True

        # First iteration
        if self.best_score is None:
            self.best_score = loss
        # Loss is better: reset the counter
        elif loss <= self.best_score - self.min_delta:
            self.best_score = loss
            self.counter = 0
        else:
            self.counter += 1

        return False




