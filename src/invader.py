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
        self.purs_num = 0
        self.KP = 10.0
        self.KD = 0.1
        
    def evade(self, pursuers, target, obstacle):
        self.num_iter += 1
        v_dir = np.zeros_like(self.position)
        if self.crashed:
            return v_dir
        p_idx = self.strategy_closest_pursuer(pursuers)
        #v_dir = self.cons_purs*self.strategy_run_away(pursuers, pursuer) + self.cons_targ*self.pursuit_pure_pursuit(target)
        #if the closest pursuer is close enough, run away from him, otherwise pursue the prime
        if p_idx != -1 and np.linalg.norm(self.position - pursuers[p_idx].position) <= 3.5:
            v_dir = self.strategy_run_away(pursuers, p_idx)
        else:
            v_dir = self.pursuit_pure_pursuit(target)
        #repulsive dirs to avoid collision with obstacle
        obs_vel = self.repulsive_force_obs(obstacle)
        #summing all vectors, making acc out of them
        v_sum = v_dir + obs_vel
        u_dir = self.KP*(v_sum - self.curr_speed) - self.KD*self.curr_speed
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
    
    def repulsive_force_obs(self, circle, coll=5.0):
        rep_dir = np.zeros_like(self.position)
        if circle is None:
            return rep_dir
        #compute the distance from self and if close enough, compute the repulsive force
        #for 2D
        if len(rep_dir) == 2:
            diff = self.position - circle.center
            dist = np.linalg.norm(diff) - self.my_rad - circle.radius
            if dist < coll and dist > 0.001:
                push_dir = diff / dist
                #hyperbolic repulsive
                magnitude = (1.0 / dist - 1.0 / coll) 
                # magnitude = (coll - dist) / coll
                rep_dir += push_dir * magnitude
        #for 3D
        else:
            diff = self.position - circle[1]
            dist = np.linalg.norm(diff) - self.my_rad - circle[2]
            if dist < coll and dist > 0.001:
                push_dir = diff / dist
                #hyperbolic repulsive
                magnitude = (1.0 / dist - 1.0 / coll) 
                # magnitude = (coll - dist) / coll
                rep_dir += push_dir * magnitude
        return rep_dir 