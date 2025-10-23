import numpy as np
from agent import Agent
from invader import Invader
from prime_unit import Prime_unit
from pursuer_states import States

class Pursuer(Agent):
    def __init__(self, position, max_acc, max_omega, num, purs_num):
        super().__init__(position, max_acc, max_omega)
        self.num_iter = 0
        self.purs_num = purs_num
        #dir par
        self.purs = 1.2
        self.form = 0.9
        self.rep = 1.0
        self.prime_rep = 2.9
        #radiuses
        self.prime_coll_r = 4.5
        self.collision_r = 2.0
        self.min_formation_r = 2.0
        self.dist_formation = np.pi
        self.formation_r = max(self.purs_num*self.dist_formation/(2*np.pi), self.min_formation_r)
        self.capture_r = 5.0
        self.capture_max = 15.0
        self.form_max = np.array([self.formation_r + 3.0])
        #other
        self.target = None
        self.num = num
        self.state = States.FORM
        #controler
        self.KP = 10.0
        self.KD = 0.1
        #list of pursuers in form state
        self.form_p = None
        
    def pursue(self, targets: list[Invader], pursuers: list[Agent], prime_unit: Prime_unit):
        #calculation of the formation
        self.form_p = [p for p in pursuers if (p.state == States.FORM and np.linalg.norm(p.position - prime_unit.position) < self.form_max[0])]
        self.formation_r = max(len(self.form_p)*self.dist_formation/(2*np.pi), self.min_formation_r)
        self.form_max[0] = self.formation_r + 3.0
        #self.num_iter += 1
        #init directions
        tar_vel = np.array([0.0, 0.0])
        form_vel = np.array([0.0, 0.0])
        #if crashed, dont move, you are supposed to be dead
        if self.state == States.CRASHED:
            return tar_vel
        #previous target captured, back to formation
        if self.target != None and self.target.captured == True:
            self.target = None
            self.state = States.FORM
        #pursuer having no target, finding target, if found, pursue it
        if self.target == None:
            if self.strategy_closest_to_self_and_unit(targets, prime_unit, pursuers):
                #tar_dir = self.pursuit_pure_pursuit(self.target)
                tar_vel = self.pursuit_constant_bearing(self.target, prime_unit)
                #tar_dir = self.pursuit_augmented_PN(self.target, debug=False)
        #pursuer having target -> pursue it
        elif self.target != None and self.target.captured == False:
            form_count = len(self.form_p)
            if np.linalg.norm(self.position - prime_unit.position) < min(self.capture_max, form_count * 3):
                #tar_dir = self.pursuit_pure_pursuit(self.target)
                tar_vel = self.pursuit_constant_bearing(self.target, prime_unit)
                #tar_dir = self.pursuit_augmented_PN(self.target, debug=False)
            else:
                self.target.pursuer = None    
        #if target dir is zero, pursuer has no target -> keep the formation
        if np.array_equal(tar_vel, form_vel):
            self.target = None
            self.state = States.FORM
            form_vel = self.attr_formation_force(prime_unit, pursuers)
        #repulsive dirs to avoid collision
        rep_vel = self.repulsive_force(pursuers, self.collision_r)
        prime_rep_vel = self.repulsive_force([prime_unit], self.prime_coll_r)
        #returning sum of those
        sum_vel = self.purs*tar_vel + self.rep*rep_vel + self.prime_rep*prime_rep_vel + self.form*form_vel
        new_acc = self.KP * (sum_vel - self.curr_speed) - self.KD * self.curr_speed
        return new_acc
    
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
    
    def strategy_closest_to_self_and_unit(self, targets: list[Invader], unit: Prime_unit, purs: list[Agent]):
        #indexes of possible targets
        t_idxs = [i for i, inv in enumerate(targets) if inv.pursuer is None or inv.pursuer.state == States.CRASHED]
        if len(t_idxs) == 0:
            return False
        #pos of possible targets, indexes and positions of possible pursuers
        t_pos = np.array([targets[i].position for i in t_idxs])
        p_idxs = [i for i, p in enumerate(purs) if p.state == States.FORM]
        p_pos = np.array([purs[i].position for i in p_idxs])
        #finding self in possible indexes, if self is not there, return
        try:
            my_id = p_idxs.index(purs.index(self))
        except ValueError:
            return False
        #mask of those possible targets that are close to unit
        near_unit = np.linalg.norm(t_pos - unit.position, axis=1) < (self.formation_r + self.capture_r)
        #if none, return
        if not np.any(near_unit):
            return False
        #the final candidates for pursuing -> have no pursuer and are close enough
        cand_t_idxs = [t_idxs[i] for i in np.nonzero(near_unit)[0]]
        cand_t_pos = t_pos[near_unit]
        #for every candidate compute the distance to all possible pursuers
        for k, targ_pos in enumerate(cand_t_pos):
            dists = np.linalg.norm(p_pos - targ_pos, axis=1)
            #if self is nearest, pursue
            nearest_p_idx = int(np.argmin(dists))
            if nearest_p_idx == my_id:
                fin_t_idx = cand_t_idxs[k]
                self.target = targets[fin_t_idx]
                self.target.pursuer = self
                self.state = States.PURSUE
                return True
        #no target is closest to self
        return False
    
    def repulsive_force(self, drones: list[Agent], coll: float):
        rep_dir = np.array([0.0, 0.0])
        #for every drone, compute the distance from self and if close enough, compute the repulsive force
        for i in range(0, len(drones)):
            dist = np.linalg.norm(self.position - drones[i].position)
            if dist < coll and not (drones[i] is self):
                rep_dir += (1/dist - 1/drones[i].position) * (self.position - drones[i].position)/(dist**2) 
        #u = self.KP * (rep_dir - self.curr_speed) - self.KD * self.curr_speed
        return rep_dir
    
    def attr_formation_force(self, unit: Prime_unit, purs: list[Agent]):
        form_ps = [p for p in purs if (p.state == States.FORM and np.linalg.norm(p.position - unit.position) < self.form_max[0])]
        n = len(form_ps)
        att_dir = np.array([0.0, 0.0])
        if n == 0:
            return att_dir
        #formation_r = max(n*self.dist_formation/(np.pi*2), self.min_formation_r)
        angle_piece = 2*np.pi / n
        my_angle = angle_piece * self.num
        form_pos = np.array([unit.position[0] + self.formation_r * np.cos(my_angle), unit.position[1] + self.formation_r * np.sin(my_angle)])
        #print(form_pos)
        att_dir = form_pos - self.position
        #print(direction)
        if np.linalg.norm(att_dir) < 1e-12:
            return np.zeros_like(att_dir)
        att_dir = att_dir / np.linalg.norm(att_dir)
        att_dir = att_dir * (np.linalg.norm(form_pos - self.position))
        #u = self.KP * (direction - self.curr_speed) - self.KD * self.curr_speed
        return att_dir
    
    def pursuit_augmented_PN(self, target: Invader, dt=0.1, N=3.0, debug=False):
        _ema_target_acc = np.zeros_like(target.curr_acc)
        #geometry
        r = target.position - self.position
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-8:
            return self.curr_speed.copy()
        r_hat = r / r_norm
        v_rel = target.curr_speed - self.curr_speed
        V_c = -np.dot(v_rel, r_hat)
        LOS_der = (v_rel - np.dot(v_rel, r_hat) * r_hat) / (r_norm + 1e-12)
        #EMA filter
        alpha = 0.2
        _ema_target_acc = alpha * target.curr_acc + (1.0 - alpha) * _ema_target_acc
        a_T = _ema_target_acc
        a_T_perp = a_T - np.dot(a_T, r_hat) * r_hat
        #PN term
        V_c_eff = V_c / (1 + np.exp(-20 * V_c))  #sigmoid
        pn_term = N * V_c_eff * LOS_der
        #debug
        if debug:
            pn_n = np.linalg.norm(pn_term)
            at_n = np.linalg.norm(a_T_perp)
            dot = np.dot(pn_term, a_T_perp)
            angle = np.degrees(np.arccos(np.clip(dot / (pn_n*at_n + 1e-12), -1, 1)))
            print(f"pn_term={pn_term}, |pn|={pn_n:.6f}; a_T_perp={a_T_perp}, |aTperp|={at_n:.6f}; angle={angle:.2f} deg; V_c={V_c:.3f}, |LOS'|={np.linalg.norm(LOS_der):.6f}")
        APN_u = pn_term + 0.3*a_T_perp
        acc_norm = np.linalg.norm(APN_u)
        if acc_norm > self.max_acc:
            APN_u = APN_u / acc_norm * self.max_acc
        v_new = self.curr_speed + APN_u * dt
        speed = np.linalg.norm(v_new)
        if speed > self.max_speed:
            v_new = v_new / speed * self.max_speed
        return v_new
        
    #TODO: deal with exceptions (e.g. a = 0)
    def pursuit_constant_bearing(self, target: Invader, unit: Prime_unit):
        #velocity vector of the target
        if np.linalg.norm(target.curr_speed) >= self.max_speed:
            return self.pursuit_pure_pursuit(target)
        v_tar = target.curr_speed #/ np.linalg.norm(target.curr_speed)) * target.max_speed
        #line of sight
        r = target.position - self.position
        #coeficients of quadratic equation
        a = np.dot(r, r)
        b = -2*np.dot(v_tar, r)
        c = np.dot(v_tar, v_tar) - self.max_speed**2
        #discriminant
        D = b**2 - 4*a*c
        CB_dir = np.array([0.0, 0.0])
        #positive D
        if D >= 1e-6:
            lambda1 = (-b + np.sqrt(D))/(2*a)
            lambda2 = (-b - np.sqrt(D))/(2*a)
            CB_dir1 = v_tar - lambda1*r
            CB_dir2 = v_tar - lambda2*r
            if np.dot(CB_dir1, r) > 0:
                CB_dir = CB_dir1
            else:
                CB_dir = CB_dir2
        #negative D
        else:
            return self.pursuit_pure_pursuit(target)
        if np.linalg.norm(CB_dir) < 1e-12:
            return np.zeros_like(CB_dir)
        CB_dir = CB_dir / np.linalg.norm(CB_dir)
        CB_dir = CB_dir * self.max_speed
        return CB_dir
    
    def pursuit_pure_pursuit(self, target: Invader):
        PP_dir = target.position - self.position
        if np.linalg.norm(PP_dir) < 1e-12:
            return np.zeros_like(PP_dir)
        PP_dir = PP_dir / np.linalg.norm(PP_dir)
        PP_dir = PP_dir * self.max_speed
        return PP_dir