import numpy as np
from agent import Agent

class Prime_unit(Agent):
    def __init__(self, position, speed, max_omega):
        super().__init__(position, speed, max_omega)
        self.finished = False
        self.took_down = False
        
    def fly(self, way_point):
        if np.sum((self.position - way_point)**2) < 0.25:
            self.finished = True
        return way_point - self.position