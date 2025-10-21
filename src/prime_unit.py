import numpy as np
from agent import Agent

class Prime_unit(Agent):
    def __init__(self, position, speed, max_omega):
        super().__init__(position, speed, max_omega)
        self.finished = False
        self.took_down = False
        #controler
        self.KP = 10.0
        self.KD = 0.1
        
    def fly(self, way_point):
        if np.sum((self.position - way_point)**2) < 0.25:
            self.finished = True
        v_dir = way_point - self.position
        u_dir = self.KP * (v_dir - self.curr_speed) - self.KD * self.curr_speed
        return u_dir