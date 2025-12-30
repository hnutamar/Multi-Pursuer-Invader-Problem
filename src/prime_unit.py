import numpy as np
from agent import Agent
from prime_mode import Modes

class Prime_unit(Agent):
    def __init__(self, position, max_acc, max_omega, my_rad):
        super().__init__(position, max_acc, max_omega)
        self.my_rad = my_rad
        #controler
        self.KP = 10.0
        self.KD = 0.1
        
        self.rep_force = self.max_speed * 1.5
        #not needed ríght now
        self.axis_a = 0
        self.axis_b = 0
        self.rot_angle = 0
        self.center = position
        
        self.t_circle = 7.0
        
    def fly(self, way_point, invaders, pursuers, mode):
        #finished, stay on this point
        if np.sum((self.position - way_point)**2) < 0.25 or self.finished:
            self.finished = True
            return np.zeros_like(way_point)
        #repulsive force against all drones
        rep_vel_i = self.repulsive_force(invaders, 3.0)
        if len(pursuers) != 0:
            rep_vel_p = self.repulsive_force(pursuers, pursuers[0].formation_r_min + 0.5)
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
        purs_vel = np.array([1*rel_pos[1] + 2*rel_pos[0]*rho, -1*rel_pos[0] + 2*rel_pos[1]*rho])
        return purs_vel
    
    def goal_force(self, way_point):
        goal_vel = way_point - self.position
        goal_speed = np.linalg.norm(goal_vel)
        if goal_speed > self.max_speed:
            goal_vel = (goal_vel / goal_speed) * self.max_speed
        return goal_vel

    def repulsive_force(self, drones: list[Agent], coll: float):
        rep_dir = np.zeros_like(self.position)
        #for every drone, compute the distance from self and if close enough, compute the repulsive force
        for drone in drones:
            if drone is self:
                continue
            diff = self.position - drone.position
            dist = np.linalg.norm(diff) - self.my_rad - drone.my_rad
            if dist < coll and dist > 0.001:
                push_dir = diff / dist
                #hyperbolic repulsive
                magnitude = (1.0 / dist - 1.0 / coll) 
                # magnitude = (coll - dist) / coll
                rep_dir += push_dir * magnitude
        return rep_dir 
