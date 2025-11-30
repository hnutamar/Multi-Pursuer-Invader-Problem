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
        self.num_iter = np.longdouble(0.0)
        #self.purs_num = purs_num
        #dir par
        self.purs = 1.2
        self.form = 0.9
        self.rep_in_form = 10.0
        self.rep_in_purs = 7.0
        #self.prime_rep_in_form = 0.0
        self.prime_rep_in_purs = 2.9
        #radiuses
        self.prime_coll_r = 4.5
        self.collision_r = 2.0
        self.vis_r = 3.0
        #self.min_formation_r = 2.0
        #self.dist_formation = np.pi/1.8
        #self.formation_r = max(self.purs_num*self.dist_formation/(2*np.pi), self.min_formation_r)
        self.formation_r = 2.5
        #print(self.formation_r)
        self.capture_r = 10.0
        self.capture_max = 20.0
        #self.form_max = np.array([self.formation_r + 3.0])
        #radiuses for target circling
        self.t_circle = 1.0
        self.target_close = 2.0
        self.safe_circle_r = 3.0
        #other
        self.target = None
        #self.num = num
        self.state = States.FORM
        #self.capture_angle = np.array([np.pi/2, ])
        #controler
        self.KP = 10.0
        self.KD = 0.1
        #list of pursuers in form state
        #self.form_p = None
        #list of targets
        self.ignored_targs = {}
        self.circle_dir = 1
        self.pred_time = 20
        self.purs_types = {"circling": 1,
                           "const_bear": 2,
                           "pure_pursuit": 3}
        self.capture_cooldown = 100
        
    def pursue(self, targets: list[Invader], pursuers: list[Agent], prime_unit: Prime_unit):
        #calculation of the formation
        #self.form_p = [p for p in pursuers if (p.state == States.FORM and np.linalg.norm(p.position - prime_unit.position) < self.form_max[0])]
        #self.formation_r = max(len(self.form_p)*self.dist_formation/(2*np.pi), self.min_formation_r)
        #self.form_max[0] = self.formation_r + 3.0
        self.num_iter += self.dt
        #init directions
        tar_vel = np.zeros_like(self.position)
        form_vel = np.zeros_like(self.position)
        #if crashed, dont move, you are supposed to be dead
        if self.state == States.CRASHED:
            return tar_vel
        #previous target captured, back to formation
        if self.target != None and self.target[0].captured == True:
            self.target = None
            self.state = States.FORM
        #pursuer having no target, finding target, if found, pursue it
        if self.target == None:
            if self.strategy_capture_cone(targets, prime_unit, self.num_iter, cooldown=self.capture_cooldown*self.dt):
                tar_vel = self.pursue_target(self.target, pursuers, prime_unit)
            elif self.strategy_target_close(targets):
                tar_vel = self.pursue_target(self.target, pursuers, prime_unit)
        #pursuer having target -> pursue it
        elif self.target != None and self.target[0].captured == False:
            if not (self.target[1] != self.purs_types["circling"] and np.linalg.norm(self.position - prime_unit.position) > self.capture_max): #min(self.capture_max, form_count * 3):
                tar_vel = self.pursue_target(self.target, pursuers, prime_unit) 
        #if target dir is zero, pursuer has no target -> keep the formation
        if np.array_equal(tar_vel, form_vel):
            self.target = None
            self.state = States.FORM
            form_vel = self.form_vortex_field(prime_unit)
        if form_vel.size != self.position.size:
            form_vel = np.append(form_vel, 0)
        if tar_vel.size != self.position.size:
            tar_vel = np.append(tar_vel, 0)
        #repulsive dirs to avoid collision
        rep_vel = self.repulsive_force(pursuers, self.collision_r)
        #returning sum of those
        if not np.array_equal(form_vel, np.zeros_like(form_vel)):
            sum_vel = self.rep_in_form*rep_vel + self.form*form_vel
        else:
            prime_rep_vel = self.repulsive_force([prime_unit], self.prime_coll_r)
            sum_vel = self.purs*tar_vel + self.rep_in_purs*rep_vel + self.prime_rep_in_purs*prime_rep_vel
        new_acc = self.KP * (sum_vel - self.curr_speed) - self.KD * self.curr_speed
        return new_acc
    
    def strategy_capture_cone(self, targets: list[Invader], unit: Prime_unit, sim_time: float, cooldown: float = 10.0):
        if not targets:
            return False
        if not hasattr(self, "ignored_targs"):
            self.ignored_targs = {}
        #invader pos
        t_pos = np.atleast_2d(np.array([i.position for i in targets]))
        #those invaders, that are close enough to unit
        near_mask = np.linalg.norm(t_pos - unit.position, axis=1) < self.capture_r
        #those invaders, that are forward to pursuer
        forward_mask = np.dot(t_pos - self.position, self.curr_speed) > 0.0
        #combo of both, choosing from these targets
        near_and_forward = np.logical_and(near_mask, forward_mask)
        #removing invaders with after cooldown
        to_remove = []
        for inv, t_last in list(self.ignored_targs.items()):
            if inv not in targets:
                to_remove.append(inv)
            else:
                dist = np.linalg.norm(inv.position - unit.position)
                if (sim_time - t_last) > cooldown and dist >= self.capture_r:
                    to_remove.append(inv)
        for inv in to_remove:
            del self.ignored_targs[inv]
        #candidates for pursuing
        cand_t_idxs = [
            i for i, inv in enumerate(targets)
            if near_and_forward[i] and inv not in self.ignored_targs
        ]
        #those behind go to ignore list, if not already there
        behind_idxs = [
            i for i, inv in enumerate(targets)
            if near_mask[i] and not forward_mask[i]
        ]
        for i in behind_idxs:
            #if not targets[i] in self.ignored_targs:
            self.ignored_targs[targets[i]] = sim_time
        #no candidate -> return
        if not cand_t_idxs:
            return False
        #choosing best candidate
        cand_t_pos = t_pos[cand_t_idxs]
        unit_vec = unit.position - self.position
        unit_dist = np.linalg.norm(unit_vec) + 1e-9
        angles = np.dot(cand_t_pos - self.position, unit_vec) / (
            np.linalg.norm(cand_t_pos - self.position, axis=1) * unit_dist
        )
        #best angles between 80 and 130
        mask = (angles >= np.cos(13*np.pi/18)) & (angles <= np.cos(4*np.pi/9))
        if np.any(mask):
            #the one closest to 120 is the best -> pursue him
            masked_angles = np.copy(angles)
            masked_angles[~mask] = np.inf
            best_local_idx = np.argmin(masked_angles)
            best_idx = cand_t_idxs[best_local_idx]
            self.target = [targets[best_idx], self.purs_types['circling']]
            self.state = States.PURSUE
            #clearing ignore list, not needed now, because pursuer has target
            self.ignored_targs.clear()
            return True
        #no target choosed, everyone go to ignore list, if not already there
        for i in cand_t_idxs:
            #if not targets[i] in self.ignored_targs:
            self.ignored_targs[targets[i]] = sim_time
        return False
    
    def strategy_target_close(self, targets: list[Invader]):
        for t in targets:
            if np.linalg.norm(t.position - self.position) < self.target_close:
                self.target = [t, self.purs_types['circling']]
                self.state = States.PURSUE
                #clearing ignore list, not needed now, because pursuer has target
                self.ignored_targs.clear()
                return True
        return False    
    
    def repulsive_force(self, drones: list[Agent], coll: float):
        rep_dir = np.zeros_like(self.position)
        #for every drone, compute the distance from self and if close enough, compute the repulsive force
        for drone in drones:
            dist = np.linalg.norm(self.position - drone.position)
            if dist < coll and not (drone is self):
                rep_dir += (1/dist - 1/drone.position) * (self.position - drone.position)/(dist**2) 
        return rep_dir

    def rotate(self, v, angle_rad):
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[c, -s],
                    [s,  c]])
        return R @ v

    def form_vortex_field(self, unit: Prime_unit):
        my_pos = self.position
        unit_pos = unit.position
        unit_vel = unit.curr_speed
        if self.position.size == 3:
            my_pos = np.delete(my_pos, -1)
            unit_pos = np.delete(unit_pos, -1)
            unit_vel = np.delete(unit_vel, -1)
        rot_angle = np.arctan2(unit_vel[1], unit_vel[0])
        axis_a = max(2.0*np.linalg.norm(unit_vel), self.formation_r)
        axis_b = max(1.3*np.linalg.norm(unit_vel), self.formation_r)
        if np.linalg.norm(unit_vel) <= 0.1:
            rel_center = np.array([0, 0])
        else:
            rel_center = np.array([-0.7*axis_a, 0])
        center = unit_pos - self.rotate(rel_center, rot_angle)
        #the center of the vortex field shifted in the current unit speed vector, because unit is moving
        # rel_pos = my_pos - (unit_pos + unit_vel * self.dt * self.pred_time)
        # rho = 1 - (rel_pos[0]**2/self.formation_r**2) - (rel_pos[1]**2/self.formation_r**2)
        rel_pos = self.rotate(my_pos - center, -rot_angle)
        rel_norm_pos = self.rotate(my_pos - unit_pos, -rot_angle)
        rho = 1 - (rel_pos[0]**2/axis_a**2) - (rel_pos[1]**2/axis_b**2)
        loc_norm = np.array([2*rel_norm_pos[0]/axis_a**2, 2*rel_norm_pos[1]/axis_b**2])
        norm = self.rotate(loc_norm, rot_angle)
        normalized = norm/np.linalg.norm(norm)
        #inside of circle
        if rho > 0:
            alpha = 7.0
        #outside of circle
        else:
            alpha = 1.0
        #circle around center
        loc_fdwrd = np.array([(-axis_a/axis_b)*rel_pos[1], (axis_b/axis_a)*rel_pos[0]])
        fdwrd = self.rotate(loc_fdwrd, rot_angle)
        fdbck = np.array([alpha*normalized[0]*rho, alpha*normalized[1]*rho])
        form_vel = self.circle_dir*fdwrd + fdbck
        #form_vel = np.array([self.circle_dir*rel_pos[1] + alpha*rel_pos[0]*rho, -self.circle_dir*rel_pos[0] + alpha*rel_pos[1]*rho])
        return form_vel
    
    def pursue_target(self, target: list[Invader, int], purs: list[Agent], unit: Prime_unit):
        #if target is faster then pursuer, just pure pursue him
        if np.linalg.norm(target[0].curr_speed) >= self.max_speed: #or target[1] != self.purs_types['circling']:
            return self.pursuit_pure_pursuit(target)
        #still too fast for encirclement, CB him
        elif np.linalg.norm(target[0].curr_speed) >= self.max_speed/2:
            return self.pursuit_constant_bearing(target)
        #if more then one is chasing him and he is further from unit, circle him
        if np.linalg.norm(unit.position - target[0].position) >= self.safe_circle_r:
            for p in purs:
                if p is not self and p.target != None and p.target[0] is target[0]:
                    return self.pursuit_circling(target)
        #no one else is chasing him, catch him
        return self.pursuit_constant_bearing(target)
    
    def pursuit_circling(self, target: list[Invader, int]):
        target[1] = self.purs_types['circling']
        #strong repulsive force is needed
        self.rep_in_purs = 7.0
        self.prime_rep_in_purs = 2.9
        #the center of the vortex field is not shifted
        rel_pos = self.position - (target[0].position) #+ target[0].curr_speed * self.dt * self.pred_time)
        rho = 1 - (rel_pos[0]**2/self.t_circle**2) - (rel_pos[1]**2/self.t_circle**2)
        #inside of circle
        if rho > 0:
            alpha = 7.0
        #outside of circle
        else:
            alpha = 1.0
        purs_vel = np.array([self.circle_dir*rel_pos[1] + alpha*rel_pos[0]*rho, -self.circle_dir*rel_pos[0] + alpha*rel_pos[1]*rho])
        # vel_norm = np.linalg.norm(purs_vel)
        # if vel_norm > 1e-8:
        #     purs_vel = purs_vel/vel_norm
        return purs_vel
        
    def pursuit_constant_bearing(self, target: list[Invader, int]):
        #target is quite fast, circling is not possible
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
        CB_dir = np.zeros_like(self.position)
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
        #norming it to the max speed
        if np.linalg.norm(CB_dir) < 1e-12:
            return np.zeros_like(CB_dir)
        CB_dir = CB_dir / np.linalg.norm(CB_dir)
        CB_dir = CB_dir * self.max_speed
        return CB_dir
    
    def pursuit_pure_pursuit(self, target: list[Invader, int]):
        #target is very fast, pure pursuit
        target[1] = self.purs_types['pure_pursuit']
        self.rep_in_purs = 1.0
        self.prime_rep_in_purs = 2.9
        PP_dir = target[0].position - self.position
        #norming it to the max speed
        if np.linalg.norm(PP_dir) < 1e-12:
            return np.zeros_like(PP_dir)
        PP_dir = PP_dir / np.linalg.norm(PP_dir)
        PP_dir = PP_dir * self.max_speed
        return PP_dir
    
    # def attr_formation_force(self, unit: Prime_unit, purs: list[Agent]):
    #     form_ps = [p for p in purs if (p.state == States.FORM and np.linalg.norm(p.position - unit.position) < self.form_max[0])]
    #     n = len(form_ps)
    #     att_dir = np.array([0.0, 0.0])
    #     if n == 0:
    #         return att_dir
    #     #formation_r = max(n*self.dist_formation/(np.pi*2), self.min_formation_r)
    #     angle_piece = 2*np.pi / n
    #     my_angle = angle_piece * self.num
    #     form_pos = np.array([unit.position[0] + self.formation_r * np.cos(my_angle), unit.position[1] + self.formation_r * np.sin(my_angle)])
    #     #print(form_pos)
    #     att_dir = form_pos - self.position
    #     #print(direction)
    #     if np.linalg.norm(att_dir) < 1e-12:
    #         return np.zeros_like(att_dir)
    #     att_dir = att_dir / np.linalg.norm(att_dir)
    #     att_dir = att_dir * (np.linalg.norm(form_pos - self.position))
    #     #u = self.KP * (direction - self.curr_speed) - self.KD * self.curr_speed
    #     return att_dir
    
    # def pursuit_augmented_PN(self, target: Invader, dt=0.1, N=3.0, debug=False):
    #     _ema_target_acc = np.zeros_like(target.curr_acc)
    #     #geometry
    #     r = target.position - self.position
    #     r_norm = np.linalg.norm(r)
    #     if r_norm < 1e-8:
    #         return self.curr_speed.copy()
    #     r_hat = r / r_norm
    #     v_rel = target.curr_speed - self.curr_speed
    #     V_c = -np.dot(v_rel, r_hat)
    #     LOS_der = (v_rel - np.dot(v_rel, r_hat) * r_hat) / (r_norm + 1e-12)
    #     #EMA filter
    #     alpha = 0.2
    #     _ema_target_acc = alpha * target.curr_acc + (1.0 - alpha) * _ema_target_acc
    #     a_T = _ema_target_acc
    #     a_T_perp = a_T - np.dot(a_T, r_hat) * r_hat
    #     #PN term
    #     V_c_eff = V_c / (1 + np.exp(-20 * V_c))  #sigmoid
    #     pn_term = N * V_c_eff * LOS_der
    #     #debug
    #     if debug:
    #         pn_n = np.linalg.norm(pn_term)
    #         at_n = np.linalg.norm(a_T_perp)
    #         dot = np.dot(pn_term, a_T_perp)
    #         angle = np.degrees(np.arccos(np.clip(dot / (pn_n*at_n + 1e-12), -1, 1)))
    #         print(f"pn_term={pn_term}, |pn|={pn_n:.6f}; a_T_perp={a_T_perp}, |aTperp|={at_n:.6f}; angle={angle:.2f} deg; V_c={V_c:.3f}, |LOS'|={np.linalg.norm(LOS_der):.6f}")
    #     APN_u = pn_term + 0.3*a_T_perp
    #     acc_norm = np.linalg.norm(APN_u)
    #     if acc_norm > self.max_acc:
    #         APN_u = APN_u / acc_norm * self.max_acc
    #     v_new = self.curr_speed + APN_u * dt
    #     speed = np.linalg.norm(v_new)
    #     if speed > self.max_speed:
    #         v_new = v_new / speed * self.max_speed
    #     return v_new
    
    # def strategy_closest_invader(self, targets: list[Invader]):
    #     #pick the closest invader
    #     poss_targs = np.array([inv.position for inv in targets])
    #     idx = -1
    #     #option so that pursuer will not target another invader till the last one is dead
    #     #if self.target in targets:
    #     #    idx = targets.index(self.target)
    #     if len(poss_targs) != 0:
    #         idx = np.argmin(np.linalg.norm(poss_targs - self.position, axis=1))
    #         self.target = [targets[idx], self.purs_types['circling']]
    #     return idx
    
    # def strategy_closest_to_prime_unit(self, targets: list[Invader], unit: Prime_unit):
    #     #pick the invader closest to prime unit
    #     poss_targs = np.array([inv.position for inv in targets])
    #     idx = -1
    #     if len(poss_targs) != 0:
    #         idx = np.argmin(np.linalg.norm(poss_targs - unit.position, axis=1))
    #         self.target = [targets[idx], self.purs_types['circling']]
    #     return idx
    
    # def strategy_combo_closest_unit_invader(self, targets: list[Invader], unit: Prime_unit):
    #     ALPHA = 1.2
    #     BETA = 0.1
    #     poss_targs = np.array([inv.position for inv in targets])
    #     idx = -1
    #     if len(poss_targs) != 0:
    #         idx = np.argmin(ALPHA * np.linalg.norm(poss_targs - unit.position, axis=1) + BETA * np.argmin(np.linalg.norm(poss_targs - self.position, axis=1)))
    #         self.target = [targets[idx], self.purs_types['circling']]
    #     return idx
    
    # def strategy_closest_to_self_and_unit(self, targets: list[Invader], unit: Prime_unit, purs: list[Agent]):
    #     #indexes of possible targets
    #     t_idxs = [i for i, inv in enumerate(targets) if inv.pursuer is None or inv.pursuer.state == States.CRASHED]
    #     if len(t_idxs) == 0:
    #         return False
    #     #pos of possible targets, indexes and positions of possible pursuers
    #     t_pos = np.array([targets[i].position for i in t_idxs])
    #     p_idxs = [i for i, p in enumerate(purs) if p.state == States.FORM]
    #     p_pos = np.array([purs[i].position for i in p_idxs])
    #     #finding self in possible indexes, if self is not there, return
    #     try:
    #         my_id = p_idxs.index(purs.index(self))
    #     except ValueError:
    #         return False
    #     #mask of those possible targets that are close to unit
    #     near_unit = np.linalg.norm(t_pos - unit.position, axis=1) < self.capture_r
    #     #if none, return
    #     if not np.any(near_unit):
    #         return False
    #     #the final candidates for pursuing -> have no pursuer and are close enough
    #     cand_t_idxs = [t_idxs[i] for i in np.nonzero(near_unit)[0]]
    #     cand_t_pos = t_pos[near_unit]
    #     #for every candidate compute the distance to all possible pursuers
    #     for k, targ_pos in enumerate(cand_t_pos):
    #         dists = np.linalg.norm(p_pos - targ_pos, axis=1)
    #         #if self is nearest, pursue
    #         nearest_p_idx = int(np.argmin(dists))
    #         if nearest_p_idx == my_id:
    #             fin_t_idx = cand_t_idxs[k]
    #             self.target = [targets[fin_t_idx], self.purs_types['circling']]
    #             self.target.pursuer = self
    #             self.state = States.PURSUE
    #             return True
    #     #no target is closest to self
    #     return False