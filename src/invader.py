import numpy as np
from agent import Agent
#from pursuer import Pursuer

class Invader(Agent):
    def __init__(self, position, speed, max_omega):
        super().__init__(position, speed, max_omega)
        self.num_iter = 0
        self.captured = False
        self.local_drone_dir = np.random.uniform(-1, 1, size=2)
        self.cons_targ = 0.5
        self.cons_purs = 0.5
        
    def evade(self, pursuers, target):
        self.num_iter += 1
        if self.captured:
            self.local_drone_dir = np.array([0, 0])
        else:
            pursuer = self.strategy_closest_pursuer(pursuers)
            #self.local_drone_dir = self.cons_purs*(self.position - pursuers[pursuer].position) + self.cons_targ*self.pursuit_pure_pursuit(target)
            if np.linalg.norm(self.position - pursuers[pursuer].position) <= 2.5:
                self.local_drone_dir = self.position - pursuers[pursuer].position
            else:
                self.local_drone_dir = self.pursuit_pure_pursuit(target)
        return self.local_drone_dir
    
    def strategy_closest_pursuer(self, puruers):
        #pick the closest invader
        poss_targs = np.array([inv.position for inv in puruers])
        idx = -1
        #option so that pursuer will not target another invader till the last one is dead
        #if self.target in targets:
        #    idx = targets.index(self.target)
        if len(poss_targs) != 0:
            idx = np.argmin(np.linalg.norm(poss_targs - self.position, axis=1))
            self.target = puruers[idx]
        return idx
    
    def movement_random(self):
        if self.num_iter % 30 == 0:
            new_dir = np.random.uniform(-1, 1, size=2)
            self.local_drone_dir = new_dir
        return
    
    def pursuit_pure_pursuit(self, target: Agent):
        return target.position - self.position