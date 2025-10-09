import numpy as np
from agent import Agent

class Invader(Agent):
    def __init__(self, position, speed):
        super().__init__(position, speed)
        self.num_iter = 0
        self.drone_dir = np.random.uniform(-1, 1, size=2)
        
    def evade(self, pursuer):
        self.num_iter += 1
        if self.num_iter % 10 == 0:
            new_dir = np.random.uniform(-1, 1, size=2)
            self.drone_dir = new_dir
        return self.drone_dir