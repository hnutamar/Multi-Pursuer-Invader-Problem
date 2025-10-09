import numpy as np
from agent import Agent

class Invader(Agent):
    def __init__(self, position, speed):
        super().__init__(position, speed)
        self.num_iter = 0
        self.captured = False
        self.drone_dir = np.random.uniform(-1, 1, size=2)
        
    def evade(self, pursuer):
        self.num_iter += 1
        if self.captured:
            self.drone_dir = np.array([0, 0])
        else:
            self.movement_random()
        return self.drone_dir
    
    def movement_random(self):
        if self.num_iter % 10 == 0:
            new_dir = np.random.uniform(-1, 1, size=2)
            self.drone_dir = new_dir
        return