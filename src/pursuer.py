import numpy as np
from agent import Agent

class Pursuer(Agent):
    def pursue(self, target):
        return target.position - self.position