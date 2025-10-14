import numpy as np
from agent import Agent
from invader import Invader
from prime_unit import Prime_unit

class Pursuer(Agent):
    def __init__(self, position, speed, max_omega, num):
        super().__init__(position, speed, max_omega)
        self.num_iter = 0
        self.purs = 1.2
        self.form = 0.5
        self.rep = 1.0
        self.prime_rep = 2.9
        self.prime_coll_r = 3.5
        self.collision_r = 2.0
        self.target = None
        self.crashed = False
        self.num = num
        self.formation_r = 2.0
        
    def pursue(self, targets: list[Invader], pursuers: list[Agent], prime_unit: Prime_unit):
        #self.rep = 1.0
        self.num_iter += 1
        #direction to the target
        tar_dir = np.array([0.0, 0.0])
        if self.crashed:
            return tar_dir
        #target = self.strategy_closest_invader(targets)
        #target = self.strategy_closest_to_prime_unit(targets, prime_unit)
        target = self.strategy_combo_closest_unit_invader(targets, prime_unit)
        if target != -1:
            tar_dir = self.pursuit_constant_bearing(targets[target], prime_unit)
        #repulsive dirs to avoid collision
        rep_dir = self.repulsive_force(pursuers, self.collision_r)
        prime_rep_dir = self.repulsive_force([prime_unit], self.prime_coll_r)
        #formation direction
        form_dir = self.attr_formation_force(prime_unit, pursuers)
        #returning sum of those
        return self.purs*tar_dir + self.rep*rep_dir + self.prime_rep*prime_rep_dir + self.form*form_dir
    
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
    
    def strategy_closest_to_prime_unit(self, targets: list[Invader], unit: Prime_unit):
        #pick the invader closest to prime unit
        poss_targs = np.array([inv.position for inv in targets])
        idx = -1
        if len(poss_targs) != 0:
            idx = np.argmin(np.linalg.norm(poss_targs - unit.position, axis=1))
            self.target = targets[idx]
        return idx
    
    def strategy_combo_closest_unit_invader(self, targets: list[Invader], unit: Prime_unit):
        ALPHA = 1.2
        BETA = 0.1
        poss_targs = np.array([inv.position for inv in targets])
        idx = -1
        if len(poss_targs) != 0:
            idx = np.argmin(ALPHA * np.linalg.norm(poss_targs - unit.position, axis=1) + BETA * np.argmin(np.linalg.norm(poss_targs - self.position, axis=1)))
            self.target = targets[idx]
        return idx
    
    def repulsive_force(self, drones: list[Agent], coll: float):
        rep_dir = np.array([0.0, 0.0])
        for i in range(0, len(drones)):
            dist = np.linalg.norm(self.position - drones[i].position)
            if dist < coll and not (drones[i] is self):
                rep_dir += (1/dist - 1/drones[i].position) * (self.position - drones[i].position)/(dist**2) 
                #if dist < 1.0:
                #    self.rep = 0.8
        #if np.linalg.norm(rep_dir) != 0:
        #    rep_dir = rep_dir / np.linalg.norm(rep_dir)
        return rep_dir
    
    def attr_formation_force(self, unit: Prime_unit, pursuers: list[Agent]):
        angle_piece = 2*np.pi / len(pursuers)
        angle = angle_piece * self.num
        form_pos = np.array([unit.position[0] + self.formation_r * np.cos(angle), unit.position[1] + self.formation_r * np.sin(angle)])
        direction = form_pos - self.position
        if np.linalg.norm(direction) != 0:
            direction = direction / np.linalg.norm(direction)
        direction = direction * (np.linalg.norm(form_pos - self.position)**2)
        return direction
        
    #TODO: deal with exceptions (e.g. a = 0)
    def pursuit_constant_bearing(self, target: Invader, unit: Prime_unit):
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
        elif D < 0:
            direc = self.pursuit_pure_pursuit(target)
        if np.linalg.norm(direc) != 0:
            direc = direc / np.linalg.norm(direc)
        direc = direc/(((1 - np.exp(-np.linalg.norm(unit.position - target.position)))))
        return direc
    
    def pursuit_pure_pursuit(self, target: Invader):
        dir = target.position - self.position
        #if np.linalg.norm(dir) != 0:
        #    dir = dir / np.linalg.norm(dir)
        return dir