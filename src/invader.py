import numpy as np
from agent import Agent

class Invader(Agent):
    def __init__(self, position, max_speed, max_acc, max_omega, my_rad, dt):
        super().__init__(position, max_speed, max_acc, max_omega, dt, my_rad)
        #number of pursuers pursuing this invader
        self.purs_num = 0
        #controller
        self.KP = 10.0
        self.KD = 0.1
        #collision parameters
        self.coll_obs = 5.0
        
    def evade(self, pursuers, target, obstacles):
        v_dir = np.zeros_like(self.position)
        if self.crashed:
            return v_dir
        self.obs_centers, self.obs_radii = obstacles
        p_idx = self.strategy_closest_pursuer(pursuers)
        #v_dir = self.cons_purs*self.strategy_run_away(pursuers, pursuer) + self.cons_targ*self.pursuit_pure_pursuit(target)
        #if the closest pursuer is close enough, run away from him, otherwise pursue the prime
        if p_idx != -1 and np.linalg.norm(self.position - pursuers[p_idx].position) - self.my_rad - pursuers[p_idx].my_rad <= 3.5:
            v_dir = self.strategy_run_away(pursuers, p_idx)
        else:
            v_dir = self.pursuit_pure_pursuit(target)
        #repulsive dirs to avoid collision with obstacle
        obs_vel = self.repulsive_force_obs(self.coll_obs)
        #obs_vel = np.zeros_like(self.position)
        #summing all vectors, making acc out of them
        v_sum = v_dir + obs_vel
        #norming speed to the possible limit
        sum_norm = np.linalg.norm(v_sum)
        if sum_norm > self.cruise_speed:
            v_sum = (v_sum/sum_norm) * self.cruise_speed
        u_dir = self.KP*(v_sum - self.curr_speed) - self.KD*self.curr_speed
        return u_dir
    
    def strategy_run_away(self, purs, idx):
        dir_ = self.position - purs[idx].position
        if np.linalg.norm(dir_) > 1e-12:
            dir_ = dir_ / np.linalg.norm(dir_)
        dir_ = dir_ * self.cruise_speed
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
        dir_ = dir_ * self.cruise_speed
        return dir_
    
    def repulsive_force_obs(self, coll):
        rep_dir = np.zeros_like(self.position)
        if self.obs_centers is None:
            return rep_dir
        #obstacle centers and radiuses
        obs_centers = self.obs_centers
        obs_radii = self.obs_radii
        #vector from self to obstacle center
        vecs_to_obs = self.position - obs_centers
        #distances center to center
        dists_center = np.linalg.norm(vecs_to_obs, axis=1)
        #distance from surface to surface
        dists_surface = dists_center - obs_radii - self.my_rad
        #mask
        mask = dists_surface < coll
        #valid data
        valid_dists_center = dists_center[mask]
        valid_dists_surface = dists_surface[mask]
        valid_diffs = vecs_to_obs[mask]
        #norm and magnitude
        push_dirs = valid_diffs / valid_dists_center[:, np.newaxis]
        magnitudes = (1.0 / valid_dists_surface) - (1.0 / coll)
        #total force
        return np.sum(push_dirs * magnitudes[:, np.newaxis], axis=0) * self.cruise_speed