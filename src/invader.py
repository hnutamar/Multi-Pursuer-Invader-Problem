import numpy as np
from agent import Agent
#from pursuer import Pursuer

class Invader(Agent):
    def __init__(self, position, max_acc, max_omega, my_rad):
        super().__init__(position, max_acc, max_omega)
        self.num_iter = 0
        self.my_rad = my_rad
        self.local_drone_dir = np.random.uniform(-1, 1, size=2)
        self.cons_targ = 0.5
        self.cons_purs = 0.5
        self.pursuer = None
        self.KP = 10.0
        self.KD = 0.1
        
    def evade(self, pursuers, target):
        self.num_iter += 1
        v_dir = np.zeros_like(self.position)
        if self.crashed:
            return v_dir
        else:
            p_idx = self.strategy_closest_pursuer(pursuers)
            #v_dir = self.cons_purs*self.strategy_run_away(pursuers, pursuer) + self.cons_targ*self.pursuit_pure_pursuit(target)
            #if the closest pursuer is close enough, run away from him, otherwise pursue the prime
            if p_idx != -1 and np.linalg.norm(self.position - pursuers[p_idx].position) <= 3.5:
                v_dir = self.strategy_run_away(pursuers, p_idx)
            else:
                v_dir = self.pursuit_pure_pursuit(target)
        u_dir = self.KP*(v_dir - self.curr_speed) - self.KD*self.curr_speed
        return u_dir
    
    def strategy_run_away(self, purs, idx):
        dir_ = self.position - purs[idx].position
        if np.linalg.norm(dir_) > 1e-12:
            dir_ = dir_ / np.linalg.norm(dir_)
        dir_ = dir_ * self.max_speed
        return dir_
    
    def strategy_closest_pursuer(self, puruers):
        #find the closest pursuer
        poss_purs = np.array([pur.position for pur in puruers])
        idx = -1
        if len(poss_purs) != 0:
            idx = np.argmin(np.linalg.norm(poss_purs - self.position, axis=1))
            self.target = puruers[idx]
        return idx
    
    def pursuit_pure_pursuit(self, target: Agent):
        #targeting the prime with pure pursuit
        dir_ = target.position - self.position
        if np.linalg.norm(dir_) > 1e-12:
            dir_ = dir_ / np.linalg.norm(dir_)
        dir_ = dir_ * self.max_speed
        return dir_