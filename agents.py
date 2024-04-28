import random
import numpy as np


class UnicycleAgent:

    def __init__(self, acc_u, acc_r, max_x, max_y, damping=0.1):
        self.heading = None
        self.pos_x = None
        self.pos_y = None
        self.vel_u = None
        self.vel_r = None
        self.acc_u = acc_u
        self.acc_r = acc_r
        self.max_x = max_x - 1
        self.max_y = max_y - 1
        self.damping = damping
        self.reset()
        return
    
    def reset(self):
        self.heading = random.random() * 2*np.pi
        self.pos_x = self.max_x // 2
        self.pos_y = self.max_y // 2
        self.vel_u = 0
        self.vel_r = 0
        return

    def step(self, dt=0.1):
        ## SAMPLE ACCELERATIONS
        acc_u = random.random() 
        acc_r = random.random() * 2 - 1

        ## UPDATE VEL
        self.vel_u += acc_u * dt - self.damping * self.vel_u 
        self.vel_r += acc_r * dt - self.damping * self.vel_r

        ## UPDATE POS
        self.pos_x += self.vel_u * np.cos(self.heading) * dt
        self.pos_y += self.vel_u * np.sin(self.heading) * dt
        self.heading += self.vel_r * dt % (2*np.pi)

        ## MAP POS TO INTEGER GRID
        grid_x = int(max(min(self.pos_x, self.max_x), 0))
        grid_y = int(max(min(self.pos_y, self.max_y), 0))

        ## BOUNCE OFF THE WALLS OF THE GRID
        if self.pos_x < 0 or self.pos_x > self.max_x:
            self.heading = (np.pi - self.heading) % (2*np.pi)
            self.pos_x = grid_x
        if self.pos_y < 0 or self.pos_y > self.max_y:
            self.heading = (-self.heading) % (2*np.pi)
            self.pos_y = grid_y

        return np.array((grid_x, grid_y))
