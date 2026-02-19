import numpy as np
from agent import Agent
from prime_mode import Modes

class Prime_unit(Agent):
    def __init__(self, position, max_acc, max_omega, my_rad, dt):
        super().__init__(position, max_acc, max_omega, dt, my_rad)
        #controler
        self.KP = 10.0
        self.KD = 0.1
        #repulsion force from other drones
        self.rep_force = 0.7
        #speed cap according to the speed of pursuers
        self.biggest_poss_speed = self.max_speed
        #for circle mode
        self.t_circle = 7.0
        
    def fly(self, way_point, invaders, pursuers, mode):
        if pursuers:
            self.biggest_poss_speed = max(np.sqrt(0.1 / self.CD), 0.15 * pursuers[0].max_speed)
        #finished, stay on this point
        if np.sum((self.position - way_point)**2) < 0.25 or self.finished:
            self.finished = True
            return np.zeros_like(way_point)
        #repulsive force against all drones
        rep_vel_i = self.repulsive_force(invaders, 3.0, False)
        rep_vel_p = np.zeros_like(self.position)
        if pursuers:
            rep_vel_p = self.repulsive_force(pursuers, pursuers[0].formation_r_min + 0.5, True)
        #direction of the goal
        if mode == Modes.CIRCLE:
            goal_vel = self.vortex_circle(way_point)
        elif mode == Modes.LINE:
            goal_vel = self.goal_force(way_point)
        #summing all the velocities
        sum_vel = goal_vel + self.rep_force * rep_vel_i + self.rep_force * rep_vel_p
        #making acceleration out of it
        sum_acc = self.KP * (sum_vel - self.curr_speed) - self.KD * self.curr_speed
        return sum_acc
    
    def vortex_circle(self, way_point):
        rel_pos = self.position - way_point
        rho = 1 - (rel_pos[0]**2/self.t_circle**2) - (rel_pos[1]**2/self.t_circle**2)
        goal_vel = np.array([1*rel_pos[1] + 1*rel_pos[0]*rho, -1*rel_pos[0] + 1*rel_pos[1]*rho])
        goal_speed = np.linalg.norm(goal_vel)
        if goal_speed > self.biggest_poss_speed :
            goal_vel = (goal_vel / goal_speed) * self.biggest_poss_speed 
        return goal_vel
    
    def goal_force(self, way_point):
        goal_vel = way_point - self.position
        goal_speed = np.linalg.norm(goal_vel)
        if goal_speed > self.biggest_poss_speed :
            goal_vel = (goal_vel / goal_speed) * self.biggest_poss_speed 
        return goal_vel
    
    def sigmoid(self, x):
        #sigmoid function
        return 1 / (1 + np.exp(-x))

    def repulsive_force(self, drones: list[Agent], coll: float, purs: bool):
        rep_dir = np.zeros_like(self.position)
        #for every drone, compute the distance from self and if close enough, compute the repulsive force
        for drone in drones:
            if drone is self:
                continue
            diff = self.position - drone.position
            dist = np.linalg.norm(diff) - self.my_rad - drone.my_rad
            if purs:
                rel_unit_pos = drone.position - self.position
                #radius is combination of two according to prime fly direction
                sigm = self.sigmoid(np.dot(self.curr_speed, rel_unit_pos))
                coll = sigm * drone.formation_r + (1 - sigm) * drone.formation_r_min + 0.1
            if dist < coll and dist > 0.001:
                push_dir = diff / dist
                #hyperbolic repulsive
                magnitude = (1.0 / dist - 1.0 / coll) 
                # magnitude = (coll - dist) / coll
                rep_dir += push_dir * magnitude
        return rep_dir 
