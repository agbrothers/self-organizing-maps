import random
import numpy as np
from tqdm import trange

from som import SelfOrganizingMap
from agents import UnicycleAgent
from plot import Anim


if __name__ == '__main__':

    ## SET SEEDS
    random.seed(42)
    np.random.seed(42)

    ## EXPERIMENT PARAMETERS
    acc_u = 5
    acc_r = 0.3
    dt = 0.5
    p = 2
    dim = 2
    d_nbhd = 8
    n_rows = 100
    n_cols = 100
    lr_init = 0.33
    lr_decay = 0.999
    timesteps = 6400
    ratio = n_rows/(2*n_cols)
    green = np.array([0,255,0])
    grid = np.ones((n_rows, n_cols,3), dtype="uint8") * 255
    anim = Anim("unicycle_walk", dpi=1000, aspect_ratio=(ratio, 1))

    ## INITIALIZE OBJECTS
    agent = UnicycleAgent(
        acc_u=acc_u,
        acc_r=acc_r,
        max_x=n_rows, 
        max_y=n_cols
    )
    som = SelfOrganizingMap(
        n_rows=n_rows,
        n_cols=n_cols,
        n_channels=dim,
        d_nbhd=d_nbhd,
        lr_init=lr_init,
        lr_decay=lr_decay,
        p=p,
    )
    
    ## LOOP
    d_max = som.distance(np.array((0,0)), np.array((n_cols,n_rows)))
    for i in trange(timesteps):
        ## STEP AGENT
        pos = agent.step(dt)

        ## UPDATE SELF-ORGANIZING MAP
        activation, idx = som(pos)

        ## PLOT RESULTS
        activation[idx] = 0
        image = anim.map_colors(-activation, d_max)
        prev_color = grid[tuple(pos)].copy()
        grid[tuple(pos)] = green
        image[idx] = green
        anim.step(np.hstack((grid,image)))
        grid[tuple(pos)] = np.maximum(prev_color-15, 0)

    anim.close()
