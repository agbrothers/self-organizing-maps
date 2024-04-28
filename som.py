import random
import numpy as np


class SelfOrganizingMap:

    def __init__(
            self, 
            n_rows,         # NUMBER OF ROWS IN THE MAP
            n_cols,         # NUMBER OF COLUMNS IN THE MAP
            n_channels,     # NUMBER OF CHANNELS IN THE MAP
            d_nbhd,         # NEIGHBORHOOD DISTANCE THRESHOLD
            p=1,            # MINKOWSKI DISTANCE PARAMETER
            lr_init=0.1,    # INITIAL LEARNING RATE
            lr_decay=0.99,  # LEARNING RATE DECAY CONSTANT
        ):
        ## NEURAL NET
        self.neurons = np.random.random((n_rows, n_cols, n_channels))
        self.neurons *= max(n_rows, n_cols)
        row_idxs = (np.tile(np.arange(n_rows), reps=(n_cols,1))).T
        col_idxs = (np.tile(np.arange(n_cols), reps=(n_rows,1)))
        self.idxs = np.stack((row_idxs, col_idxs), axis=-1)
        
        ## LEARNING PARAMETERS
        self.lr_decay = lr_decay
        self.lr_init = lr_init
        self.t = 0
        
        ## NEIGHBORHOOD PARAMETERS
        self.d_nbhd = d_nbhd
        self.p = p
        return

    def distance(self, x, y):
        ## MINKOWSKI DISTANCE
        return np.power(
            np.power(np.abs(x-y), self.p).sum(axis=-1), 
            1/self.p
        )

    def forward(self, x):
        ## WINNER TAKES ALL RESPONSE
        d_activation = self.distance(self.neurons, x)
        winner_idx = np.unravel_index(np.argmin(d_activation), d_activation.shape)

        ## MAKE NEIGHBORS MORE SIMILAR TO THE ACTIVATION
        lr = self.lr_init * self.lr_decay**self.t
        d_map = self.distance(self.idxs, winner_idx)
        nbhd = (d_map <= self.d_nbhd)[..., None]
        delta = x - self.neurons

        ## UPDATE NEURONS
        self.neurons += lr * nbhd * delta
        self.t += 1
        return d_activation, winner_idx

    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    p = 2
    dim = 2
    n_rows = 100
    n_cols = 100
    d_nbhd = 10
    lr_init = 0.33
    lr_decay = 0.999

    som = SelfOrganizingMap(
        n_rows=n_rows,
        n_cols=n_cols,
        n_channels=dim,
        d_nbhd=d_nbhd,
        lr_init=lr_init,
        lr_decay=lr_decay,
        p=p,
    )
    pos = np.array([1,1])
    activation, idx = som(pos)
