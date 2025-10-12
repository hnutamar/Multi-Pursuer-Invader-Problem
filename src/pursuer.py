import numpy as np
from agent import Agent
from invader import Invader
from prime_unit import Prime_unit

class Pursuer(Agent):
    def __init__(self, position, speed):
        super().__init__(position, speed)
        self.num_iter = 0
        self.attr = 0.5
        self.rep = 0.1
        self.prime_rep = 0.7
        self.prime_coll_r = 2.5
        self.collision_r = 1.0
        self.target = None
        #self.captured = False
        #self.drone_dir = np.random.uniform(-1, 1, size=2)
        
    def pursue(self, targets: list[Invader], pursuers: list[Agent], prime_unit: Prime_unit):
        self.num_iter += 1
        #direction to the target
        tar_dir = np.array([0.0, 0.0])
        target = self.strategy_closest_invader(targets)
        if target != -1:
            tar_dir = self.pursuit_constant_bearing(targets[target])
        #repulsive dirs to avoid collision
        rep_dir = self.repulsive_force(pursuers, self.collision_r)
        prime_rep_dir = self.repulsive_force([prime_unit], self.prime_coll_r)
        #returning sum of those
        return self.attr * tar_dir + self.rep * rep_dir + self.prime_rep * prime_rep_dir
    
    def strategy_closest_invader(self, targets: list[Invader]):
        #pick the closest invader
        poss_targs = np.array([inv.position for inv in targets])
        idx = -1
        #option so that pursuer will not target another invader till the last one is dead
        #if self.target in targets:
        #    idx = targets.index(self.target)
        if len(poss_targs) != 0:
            idx = np.argmin(np.linalg.norm(poss_targs - self.position, axis=1))
            self.target = targets[idx]
        return idx
    
    def pursuit_pure_pursuit(self, target: Invader):
        return target.position - self.position
    
    def repulsive_force(self, drones: list[Agent], coll: float):
        rep_dir = np.array([0.0, 0.0])
        for i in range(0, len(drones)):
            dist = np.linalg.norm(self.position - drones[i].position)
            if dist < coll and not (drones[i] is self):
                rep_dir += (1/dist - 1/drones[i].position) * (self.position - drones[i].position)/(dist)**2 
        return rep_dir
        
    #TODO: deal with exceptions (e.g. a = 0)
    def pursuit_constant_bearing(self, target: Invader):
        #velocity vector of the target
        if target.speed >= self.speed:
            direc = self.pursuit_pure_pursuit(target)
            return direc
        v_tar = target.speed * (target.drone_dir / np.linalg.norm(target.drone_dir))
        #line of sight
        r = target.position - self.position
        #coeficients of quadratic equation
        a = np.dot(r, r)
        b = -2*np.dot(v_tar, r)
        c = np.dot(v_tar, v_tar) - self.speed**2
        #discriminant
        D = b**2 - 4*a*c
        direc = np.array([0.0, 0.0])
        if D >= 0:
            lambda1 = (-b + np.sqrt(D))/(2*a)
            lambda2 = (-b - np.sqrt(D))/(2*a)
            direc1 = v_tar - lambda1*r
            direc2 = v_tar - lambda2*r
            if np.dot(direc1, r) > 0:
                direc = direc1
            else:
                direc = direc2
        if D < 0:
            direc = self.pursuit_pure_pursuit(target)
        return direc