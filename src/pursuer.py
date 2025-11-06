import numpy as np
import random
import time
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
        self.rep_in_form = 7.0
        self.rep_in_purs = 7.0
        self.prime_rep_in_form = 2.9
        self.prime_rep_in_purs = 2.9
        #radiuses
        self.prime_coll_r = 4.5
        self.collision_r = 2.0
        self.min_formation_r = 2.0
        self.dist_formation = np.pi/1.8
        self.formation_r = max(self.purs_num*self.dist_formation/(2*np.pi), self.min_formation_r)
        self.capture_r = 5.0
        self.capture_max = 15.0
        self.form_max = np.array([self.formation_r + 3.0])
        #radiuses for target circling
        self.target_r = 1.5
        #other
        self.target = None
        self.num = num
        self.state = States.FORM
        self.capture_angle = np.pi/3
        #controler
        self.KP = 10.0
        self.KD = 0.1
        #list of pursuers in form state
        self.form_p = None
        #list of targets
        self.targets = {}
        self.circle_dir = 1
        self.pred_time = 20
        self.purs_types = {"circling": 1,
                           "const_bear": 2,
                           "pure_pursuit": 3}
        
    def pursue(self, targets: list[Invader], pursuers: list[Agent], prime_unit: Prime_unit):
        #calculation of the formation
        #self.form_p = [p for p in pursuers if (p.state == States.FORM and np.linalg.norm(p.position - prime_unit.position) < self.form_max[0])]
        #self.formation_r = max(len(self.form_p)*self.dist_formation/(2*np.pi), self.min_formation_r)
        #self.form_max[0] = self.formation_r + 3.0
        #self.num_iter += 1
        #init directions
        tar_vel = np.array([0.0, 0.0])
        form_vel = np.array([0.0, 0.0])
        #if crashed, dont move, you are supposed to be dead
        if self.state == States.CRASHED:
            return tar_vel
        #previous target captured, back to formation
        if self.target != None and self.target[0].captured == True:
            self.target = None
            self.state = States.FORM
        #pursuer having no target, finding target, if found, pursue it
        if self.target == None:
            if self.strategy_close_to_unit(targets, prime_unit):
                #tar_dir = self.pursuit_pure_pursuit(self.target)
                #tar_vel = self.pursuit_constant_bearing(self.target, prime_unit)
                tar_vel = self.pursuit_circling(self.target)
                #tar_dir = self.pursuit_augmented_PN(self.target, debug=False)
        #pursuer having target -> pursue it
        elif self.target != None and self.target[0].captured == False:
            #form_count = len(self.form_p)
            if np.linalg.norm(self.position - prime_unit.position) < self.capture_max: #min(self.capture_max, form_count * 3):
                #tar_dir = self.pursuit_pure_pursuit(self.target)
                #tar_vel = self.pursuit_constant_bearing(self.target, prime_unit)
                tar_vel = self.pursuit_circling(self.target)
                #tar_dir = self.pursuit_augmented_PN(self.target, debug=False)
            else:
                self.target[0].pursuer = None    
        #if target dir is zero, pursuer has no target -> keep the formation
        if np.array_equal(tar_vel, form_vel):
            self.target = None
            self.state = States.FORM
            #form_vel = self.attr_formation_force(prime_unit, pursuers)
            form_vel = self.form_vortex_field(prime_unit)
        #repulsive dirs to avoid collision
        rep_vel = self.repulsive_force(pursuers, self.collision_r)
        prime_rep_vel = self.repulsive_force([prime_unit], self.prime_coll_r)
        #returning sum of those
        if not np.array_equal(form_vel, np.zeros_like(form_vel)):
            sum_vel = self.rep_in_form*rep_vel + self.form*form_vel + self.prime_rep_in_form*prime_rep_vel
        else:
            sum_vel = self.purs*tar_vel + self.rep_in_purs*rep_vel + self.prime_rep_in_purs*prime_rep_vel
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
            self.target = [targets[idx], self.purs_types['circling']]
        return idx
    
    def strategy_closest_to_prime_unit(self, targets: list[Invader], unit: Prime_unit):
        #pick the invader closest to prime unit
        poss_targs = np.array([inv.position for inv in targets])
        idx = -1
        if len(poss_targs) != 0:
            idx = np.argmin(np.linalg.norm(poss_targs - unit.position, axis=1))
            self.target = [targets[idx], self.purs_types['circling']]
        return idx
    
    def strategy_combo_closest_unit_invader(self, targets: list[Invader], unit: Prime_unit):
        ALPHA = 1.2
        BETA = 0.1
        poss_targs = np.array([inv.position for inv in targets])
        idx = -1
        if len(poss_targs) != 0:
            idx = np.argmin(ALPHA * np.linalg.norm(poss_targs - unit.position, axis=1) + BETA * np.argmin(np.linalg.norm(poss_targs - self.position, axis=1)))
            self.target = [targets[idx], self.purs_types['circling']]
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
                self.target = [targets[fin_t_idx], self.purs_types['circling']]
                self.target.pursuer = self
                self.state = States.PURSUE
                return True
        #no target is closest to self
        return False
    
    
    def strategy_close_to_unit(self, targets: list[Invader], unit: Prime_unit, cooldown=1.0):
        if not targets:
            return False
        # pozice invaderů
        t_pos = np.atleast_2d(np.array([i.position for i in targets]))
        near_unit = np.linalg.norm(t_pos - unit.position, axis=1) < (self.formation_r + self.capture_r)
        now = time.time()
        # 1️⃣ aktualizace seznamu ignorovaných
        # - udržujeme jen ty, kteří jsou stále blízko, nebo jejich cooldown ještě neskončil
        to_remove = []
        for inv, t_last in list(self.targets.items()):
            if inv not in targets:
                to_remove.append(inv)
            else:
                dist = np.linalg.norm(inv.position - unit.position)
                if dist >= (self.formation_r + self.capture_r) and (now - t_last) > cooldown:
                    to_remove.append(inv)
        for inv in to_remove:
            del self.targets[inv]
        # 2️⃣ kandidáti: všichni, kteří jsou blízko a nejsou v cooldown zóně
        cand_t_idxs = [
            i for i, inv in enumerate(targets)
            if near_unit[i] and inv not in self.targets
        ]
        if not cand_t_idxs:
            return False
        # 3️⃣ výběr nejlepšího kandidáta podle úhlu
        cand_t_pos = t_pos[cand_t_idxs]
        unit_vec = unit.position - self.position
        unit_dist = np.linalg.norm(unit_vec)
        angles = np.dot(cand_t_pos - self.position, unit_vec) / (
            np.linalg.norm(cand_t_pos - self.position, axis=1) * unit_dist
        )
        best_local_idx = np.argmin(angles)
        best_idx = cand_t_idxs[best_local_idx]
        # 4️⃣ kontrola capture kuželu
        if angles[best_local_idx] < -np.cos(self.capture_angle):
            self.target = [targets[best_idx], self.purs_types['circling']]
            self.state = States.PURSUE
            # 🔴 ostatní invadery (v kuželu, ale nevybrané) přidej do ignorovaných s časovou známkou
            for i in cand_t_idxs:
                if i != best_idx:
                    self.targets[targets[i]] = now
            return True
        # pokud žádný nevyhovuje, aktualizuj čas pro všechny blízké invadery
        for i in cand_t_idxs:
            self.targets[targets[i]] = now
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
    
    def form_vortex_field(self, unit: Prime_unit):
        #the center of the vortex field shifted in the current unit speed vector, because unit is moving
        rel_pos = self.position - (unit.position + unit.curr_speed * self.dt * self.pred_time)
        rho = (1 - (rel_pos[0]**2 + rel_pos[1]**2))/self.formation_r**2
        purs_vel = np.array([self.circle_dir*rel_pos[1] + 1*rel_pos[0]*rho, -self.circle_dir*rel_pos[0] + 1*rel_pos[1]*rho])
        # vel_norm = np.linalg.norm(purs_vel)
        # if vel_norm > 1e-8:
        #     purs_vel = purs_vel/vel_norm
        return purs_vel
    
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
    
    def pursuit_circling(self, target: list[Invader, int]):
        if np.linalg.norm(target[0].curr_speed) >= self.max_speed/2 or target[1] != self.purs_types['circling']:
            return self.pursuit_constant_bearing(target)
        self.rep_in_purs = 7.0
        self.prime_rep_in_purs = 2.9
        #the center of the vortex field shifted in the current speed vector, because target is moving
        rel_pos = self.position - target[0].position #+ target.curr_speed * self.dt * self.pred_time)
        rho = (1 - (rel_pos[0]**2 + rel_pos[1]**2))/self.target_r**2
        purs_vel = np.array([self.circle_dir*rel_pos[1] + 1*rel_pos[0]*rho, -self.circle_dir*rel_pos[0] + 1*rel_pos[1]*rho])
        # vel_norm = np.linalg.norm(purs_vel)
        # if vel_norm > 1e-8:
        #     purs_vel = purs_vel/vel_norm
        return purs_vel
        
    #TODO: deal with exceptions (e.g. a = 0)
    def pursuit_constant_bearing(self, target: list[Invader, int]):
        #velocity vector of the target
        if np.linalg.norm(target[0].curr_speed) >= self.max_speed or target[1] == self.purs_types['pure_pursuit']:
            return self.pursuit_pure_pursuit(target)
        target[1] = self.purs_types['const_bear']
        self.rep_in_purs = 1.0
        self.prime_rep_in_purs = 2.9
        v_tar = target[0].curr_speed #/ np.linalg.norm(target.curr_speed)) * target.max_speed
        #line of sight
        r = target[0].position - self.position
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
    
    def pursuit_pure_pursuit(self, target: list[Invader, int]):
        target[1] = self.purs_types['pure_pursuit']
        self.rep_in_purs = 1.0
        self.prime_rep_in_purs = 2.9
        PP_dir = target[0].position - self.position
        if np.linalg.norm(PP_dir) < 1e-12:
            return np.zeros_like(PP_dir)
        PP_dir = PP_dir / np.linalg.norm(PP_dir)
        PP_dir = PP_dir * self.max_speed
        return PP_dir