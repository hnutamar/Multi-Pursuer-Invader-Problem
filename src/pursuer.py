import numpy as np
from agent import Agent

class Pursuer(Agent):
    def pursue(self, targets):
        direction = np.random.uniform(-1, 1, size=2)
        if targets.size != 0:
            target = self.strategy_closest_invader(targets)
            direction = self.pursuit_pure_pursuit(targets[target])
        return direction
    
    def strategy_closest_invader(self, targets):
        return np.argmin(np.linalg.norm(targets - self.position, axis=1))
    
    def pursuit_pure_pursuit(self, target_pos):
        return target_pos - self.position
    
    #TODO
    def pursuit_constant_bearing(self, target_pos):
        return