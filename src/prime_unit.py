import numpy as np
from agent import Agent

class Prime_unit(Agent):
    def __init__(self, position, max_acc, max_omega):
        super().__init__(position, max_acc, max_omega)
        self.finished = False
        self.took_down = False
        #controler
        self.KP = 10.0
        self.KD = 0.1
        
        self.rep_force = 2.0
        
    def fly(self, way_point, invaders, pursuers):
        if np.sum((self.position - way_point)**2) < 0.25:
            self.finished = True
        rep_vel_i = self.repulsive_force(invaders, 3.0)
        rep_vel_p = self.repulsive_force(pursuers, 1.0)
        goal_vel = self.goal_force(way_point)
        sum_vel = goal_vel + rep_vel_i + rep_vel_p
        sum_acc = self.KP * (sum_vel - self.curr_speed) - self.KD * self.curr_speed
        return sum_acc
    
    def goal_force(self, way_point):
        goal_vel = way_point - self.position
        goal_speed = np.linalg.norm(goal_vel)
        if goal_speed > self.max_speed:
            goal_vel = (goal_vel / goal_speed) * self.max_speed
        return goal_vel
    
    def repulsive_force(self, drones: list[Agent], coll: float):
        rep_dir = np.array([0.0, 0.0])
        #for every drone, compute the distance from self and if close enough, compute the repulsive force
        for i in range(0, len(drones)):
            dist = np.linalg.norm(self.position - drones[i].position)
            if dist < coll and not (drones[i] is self):
                rep_dir += (1/dist - 1/drones[i].position) * (self.position - drones[i].position)/(dist**2) 
        #u = self.KP * (rep_dir - self.curr_speed) - self.KD * self.curr_speed
        return rep_dir