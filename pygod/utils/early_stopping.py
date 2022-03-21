# -*- coding: utf-8 -*-
"""
Early Stopping Counter
Adapted from DGL
"""
# Author: Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause

import torch


class EarlyStopping:
    r"""Early Stopping Counter

    Parameters
    ----------
    patience : int
        The epoch number waiting after the highest score
        Default: 10
    verbose : bool
        Whether to print information
        Default: False

    """
    def __init__(self,
                 patience: int = 10,
                 verbose: bool = True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.stop = False

    def step(self, score: float, model: torch.nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            torch.save(model.state_dict(), 'es_checkpoint.pt')
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'Early Stopping Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = score
            torch.save(model.state_dict(), 'es_checkpoint.pt')
            self.counter = 0
        return self.stop
