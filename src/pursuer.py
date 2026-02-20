import numpy as np
from agent import Agent
from invader import Invader
from prime_unit import Prime_unit
from pursuer_states import States

class Pursuer(Agent):
    def __init__(self, position, max_speed, max_acc, max_omega, my_rad, purs_num, purs_vis, dt):
        super().__init__(position, max_speed, max_acc, max_omega, dt, my_rad, num_iter=purs_num)
        #visibility range
        self.vis_range = purs_vis
        #repulsive forces
        self.purs = 1.2
        self.form = 0.9
        self.rep_in_form = 5.0
        self.rep_in_purs = 6.0
        self.rep_obs = 1.0
        self.rep_invs = 20.0
        self.prime_rep_in_purs = 2.9
        #collision radiuses
        self.prime_coll_r = 10.5
        self.collision_r = min(purs_vis, 2.0)
        #formation radiuses
        self.formation_r = 2.0
        self.formation_r_min = 1.0
        #formation direction
        self.circle_dir = 1
        self.circle_dir_obs = 1
        #capture radiuses
        self.capture_r = 20.0
        self.capture_max = 30.0
        #radiuses for target circling
        self.t_circle = 1.0
        self.target_close = 2.0
        self.safe_circle_r = 3.0
        self.rep_invs_r = 8.0
        self.circle_tan_max = 3.0
        #obstacle radiuses
        self.obs_rad = 8.0
        self.rep_obs_r = 8.0
        #states
        self.target = None
        self.state = States.FORM
        #controler
        self.KP = 5.0
        self.KD = 0.1
        #list of targets
        self.ignored_targs = {}
        #self.pred_time = 20
        #capture params
        self.purs_types = {"circling": 1,
                           "const_bear1": 2,
                           "pure_pursuit1": 3,
                           "const_bear": 4,
                           "pure_pursuit": 5}
        self.capture_cooldown = 100
        if len(self.position) == 3:
            self.MAX_PURSUERS = 10
        else:
            self.MAX_PURSUERS = 4
        #params for obstacle avoidance
        self.avoid_axis = None
        self.axis_found = False
        self.curr_obs = None
        #obstacles centers and radiuses
        self.obs_centers = None
        self.obs_radii = None
        self.coll_obs = 5.0
        self.max_form_speed = 3.0
        
    def pursue(self, targets: list[Invader], pursuers: list[Agent], prime_unit: Prime_unit, precalc_data):
        #precalculated data, faster this way
        self.all_inv_pos, self.all_inv_purs_num, self.all_inv_rads, all_purs_pos, all_purs_rads, self.my_index, self.obs_centers, self.obs_radii = precalc_data
        self.all_purs_pos, self.all_purs_rads = self.get_visible_neighbors_data(all_purs_pos, all_purs_rads)
        #init directions
        tar_vel = np.zeros_like(self.position)
        form_vel = np.zeros_like(self.position)
        #if crashed, dont move, you are supposed to be dead
        if self.crashed:
            return tar_vel
        #computing closest obstacle
        if self.obs_centers is not None:
            #searching for closest obstacle
            curr_obs = self.get_nearest_obstacle(prime_unit)
            if curr_obs is None:
                self.curr_obs = None
            elif self.curr_obs is None or not np.array_equal(curr_obs['center'], self.curr_obs['center']):
                self.curr_obs = curr_obs
                self.avoid_axis = None
                self.axis_found = False
        #previous target captured, back to formation
        if self.target != None and self.target[0].crashed == True:
            self.target = None
            self.state = States.FORM
        #pursuer having no target, finding target, if found, pursue it
        if self.target == None:
            if self.strategy_capture_cone(targets, prime_unit, self.num_iter, cooldown=self.capture_cooldown*self.dt):
                tar_vel = self.pursue_target(self.target, pursuers, prime_unit)
            elif self.strategy_target_close(targets):
                tar_vel = self.pursue_target(self.target, pursuers, prime_unit)
        #pursuer having target -> pursue it
        elif self.target != None and self.target[0].crashed == False:
            if not (self.target[1] != self.purs_types["circling"] and np.linalg.norm(self.position - prime_unit.position) > self.capture_max):
                tar_vel = self.pursue_target(self.target, pursuers, prime_unit) 
        #if target dir is zero, pursuer has no target -> keep the formation
        if np.array_equal(tar_vel, form_vel):
            self.target = None
            self.state = States.FORM
            if len(self.position) == 2:
                form_vel = self.form_vortex_field_circle(prime_unit, obstacle=self.curr_obs)
            else:
                form_vel = self.form_vortex_field_sphere(prime_unit, close_purs=pursuers, obstacle=self.curr_obs)
        #repulsive dirs to avoid collision
        rep_vel = self.repulsive_force_purs(self.collision_r)
        #repulsive dirs to avoid collision with obstacle
        obs_vel = self.repulsive_force_obs(self.coll_obs)
        #returning sum of those
        if not np.array_equal(form_vel, np.zeros_like(form_vel)):
            #pushing whole formation away from invaders and obstacle
            invs_rep_vel = self.repulsive_inv_force(prime_unit, targets)
            sum_vel = self.rep_in_form*rep_vel + self.form*form_vel + self.rep_obs*obs_vel + self.rep_invs*invs_rep_vel
        else:
            prime_rep_vel = self.repulsive_force_prime(prime_unit, self.prime_coll_r)
            sum_vel = self.purs*tar_vel + self.rep_in_purs*rep_vel + self.prime_rep_in_purs*prime_rep_vel + self.rep_obs*obs_vel
        #norming speed to the possible limit
        sum_norm = np.linalg.norm(sum_vel)
        if sum_norm > self.cruise_speed:
            sum_vel = (sum_vel/sum_norm) * self.cruise_speed
        #converting to acceleration
        new_acc = self.KP * (sum_vel - self.curr_speed) - self.KD * self.curr_speed
        return new_acc
    
    def get_visible_neighbors_data(self, all_pos, all_rad):
        #dists to all
        diffs = all_pos - self.position
        dists_center = np.linalg.norm(diffs, axis=1)
        #trick, my distance is infinity
        dists_center[self.my_index] = np.inf
        #mask visibility
        dists_surface = dists_center - self.my_rad - all_rad
        mask = dists_surface < self.vis_range
        #only those visible
        if not np.any(mask):
             return np.empty((0, 3)), np.empty((0,))
        visible_pos = all_pos[mask]
        visible_rad = all_rad[mask]
        return visible_pos, visible_rad
    
    def get_nearest_obstacle(self, prime):
        #obstacle centers and radiuses
        obs_centers = self.obs_centers
        obs_radii = self.obs_radii
        #vector from prime to obstacle center
        vecs_to_obs = obs_centers - prime.position
        #distances center to center
        dists_center = np.linalg.norm(vecs_to_obs, axis=1)
        #distance from surface to surface
        dists_surface = dists_center - obs_radii - prime.my_rad
        #speed of prime
        speed_norm = np.linalg.norm(prime.curr_speed)
        #if prime is moving, interesting are only those obstacles in a way
        if speed_norm > 0.1:
            #only those who has positive dot product
            dot_products = np.sum(prime.curr_speed * vecs_to_obs, axis=1)
            mask = dot_products > 0
            #if there is none, not important
            if not np.any(mask):
                return None
            #relevant obstacles
            relevant_indices = np.where(mask)[0]
            relevant_dists = dists_surface[mask]
            #minimum
            min_idx_local = np.argmin(relevant_dists)
            real_idx = relevant_indices[min_idx_local]
            obstacle = {'center': self.obs_centers[real_idx], 'radius': self.obs_radii[real_idx]}
            return obstacle
        else:
            #just the closest
            min_idx = np.argmin(dists_surface)
            obstacle = {'center': self.obs_centers[min_idx], 'radius': self.obs_radii[min_idx]}
            return obstacle  
             
    def get_avoidance_direction(self, obstacle_pos, obstacle_rad, prime):
        #if prime is too close to obstacle, calculate which direction is better for avoidance, in 2D
        if np.linalg.norm(obstacle_pos - prime.position) - prime.my_rad - obstacle_rad <= self.obs_rad:
            if self.axis_found:
                return True
            vel = prime.curr_speed[:2]
            if np.linalg.norm(vel) < 0.1:
                return
            vec_to_obs = (obstacle_pos - prime.position)[:2]
            #2D cross product
            cross_z = vel[0] * vec_to_obs[1] - vel[1] * vec_to_obs[0]
            if cross_z > 0:
                self.axis_found = True
                self.circle_dir_obs = -1
            else:
                self.axis_found = True
                self.circle_dir_obs = 1
            return True
        return False
    
    def strategy_capture_cone(self, targets: list[Invader], unit: Prime_unit, sim_time: float, cooldown: float = 10.0):
        if not targets:
            return False
        if not hasattr(self, "ignored_targs"):
            self.ignored_targs = {}
        #invader pos
        t_pos = self.all_inv_pos
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
            and inv.purs_num < self.MAX_PURSUERS
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
        if len(targets) == 0:
            return False
        #positions, pursuer nums and radii
        t_pos = self.all_inv_pos
        t_purs_num = self.all_inv_purs_num
        t_rads = self.all_inv_rads
        #dists center to center
        dists_center = np.linalg.norm(t_pos - self.position, axis=1)
        #dists surface to surface
        dists_surface = dists_center - self.my_rad - t_rads
        #mask, only those close enough with a few pursuers
        mask = (dists_surface < self.target_close) & (t_purs_num < self.MAX_PURSUERS)
        #choosing target
        if np.any(mask):
            #first one (faster)
            idx = np.argmax(mask)
            #closest one
            # valid_indices = np.where(mask)[0]
            # local_min_idx = np.argmin(dists_surface[valid_indices])
            # idx = valid_indices[local_min_idx]
            self.target = [targets[idx], self.purs_types['circling']]
            self.state = States.PURSUE
            self.ignored_targs.clear()
            return True
        return False 
    
    def repulsive_force_purs(self, coll: float):
        if len(self.all_purs_pos) == 0:
            return np.zeros_like(self.position)
        #dists center to center
        diffs = self.position - self.all_purs_pos
        dists_center = np.linalg.norm(diffs, axis=1)
        #dists of surfaces
        dists_surface = dists_center - self.my_rad - self.all_purs_rads
        #masking relevant
        mask = (dists_surface < coll) & (dists_surface > 0.001)
        if not np.any(mask):
            return np.zeros_like(self.position)
        #valid pursuer
        valid_diffs = diffs[mask]
        valid_dists_center = dists_center[mask]
        valid_dists_surface = dists_surface[mask]
        #norm and magnitude
        push_dirs = valid_diffs / valid_dists_center[:, np.newaxis]
        magnitudes = (1.0 / valid_dists_surface) - (1.0 / coll)
        #total force
        return np.sum(push_dirs * magnitudes[:, np.newaxis], axis=0)

    def repulsive_force_prime(self, prime: Prime_unit, coll: float):
        total_force = np.zeros_like(self.position)
        #distance center to center
        diff = self.position - prime.position
        dist_center = np.linalg.norm(diff)
        if dist_center < 1e-6:
            dist_center = 1e-6
        #distance of surfaces
        dist_surface = dist_center - self.my_rad - prime.my_rad
        #computing the force, if close enough
        if dist_surface < coll:
            push_dir = diff/dist_center
            magnitude = (1.0 / dist_surface) - (1.0 / coll)
            total_force = push_dir * magnitude
        return total_force
    
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
        return np.sum(push_dirs * magnitudes[:, np.newaxis], axis=0)
    
    def repulsive_inv_force(self, prime, targets):
        rep_dir = np.zeros_like(self.position)
        close_targs = []
        dists = []
        #targets that are close
        if targets:
            t_pos = self.all_inv_pos
            t_rad = self.all_inv_rads
            #dists center to center
            vecs_to_t = t_pos - prime.position
            dists_c = np.linalg.norm(vecs_to_t, axis=1)
            #zero div protection
            dists_c[dists_c < 1e-6] = 1e-6
            #dists surface to surface
            dists_s = dists_c - t_rad - prime.my_rad
            #mask
            mask = dists_s < self.rep_invs_r
            if np.any(mask):
                #relevant data
                valid_pos = t_pos[mask]
                valid_rad = t_rad[mask]
                valid_vecs = vecs_to_t[mask]
                valid_dists_c = dists_c[mask]
                valid_dists_s = dists_s[mask]
                #point on the surface of invader
                dirs_norm = valid_vecs / valid_dists_c[:, np.newaxis]
                surface_points = valid_pos - (dirs_norm * valid_rad[:, np.newaxis])
                #storing
                close_targs.extend(surface_points)
                dists.extend(valid_dists_s)
        #obstacle dist
        if self.obs_centers is not None:
            obs_centers = self.obs_centers
            obs_radii = self.obs_radii
            #from prime to obstacle center
            vecs_to_centers = obs_centers - prime.position
            #distance center to center
            dists_center = np.linalg.norm(vecs_to_centers, axis=1)
            #protection of not devide by zero
            dists_center[dists_center < 1e-6] = 1e-6
            #distance surface to surface
            dists_surface = dists_center - obs_radii - prime.my_rad
            #only those close enough
            mask = dists_surface < self.rep_obs_r
            if np.any(mask):
                #relevant data
                valid_centers = obs_centers[mask]
                valid_radii = obs_radii[mask]
                valid_vecs = vecs_to_centers[mask]
                valid_dists_c = dists_center[mask]
                valid_dists_s = dists_surface[mask]
                #point on surface
                dirs_normalized = valid_vecs / valid_dists_c[:, np.newaxis]
                surface_points = valid_centers - (dirs_normalized * valid_radii[:, np.newaxis])
                #storing the results
                close_targs.extend(surface_points)
                dists.extend(valid_dists_s / 4.0)
        #making weighted center of mass and then repulsive force against
        if close_targs:
            close_targs = np.array(close_targs)
            dists = np.array(dists)
            #weights, the closer, the bigger
            exp_dists = np.exp(-dists)
            weights = exp_dists / np.sum(exp_dists)
            w_center_of_mass = np.sum(close_targs * weights[:, np.newaxis], axis=0)
            #calculation rel pos of pursuer, finding out if pursuer is in between prime and center of mass
            vec_prime_to_pursuer = self.position - prime.position
            vec_prime_to_com = w_center_of_mass - prime.position
            norm_p2u = np.linalg.norm(vec_prime_to_pursuer)
            norm_u2com = np.linalg.norm(vec_prime_to_com)
            if norm_p2u > 1e-6 and norm_u2com > 1e-6 and norm_p2u < norm_u2com:
                vec_a = vec_prime_to_pursuer / norm_p2u
                vec_b = vec_prime_to_com / norm_u2com
                dot_prod = np.dot(vec_a, vec_b)
                #45 degree threshold
                THRESHOLD_45_DEG = 0.7071
                if dot_prod < THRESHOLD_45_DEG:
                    return rep_dir
            else:
                return rep_dir
            #calculating rep dir
            diff = prime.position - w_center_of_mass
            dist = np.linalg.norm(diff)
            if dist > 0.001:
                push_dir = diff / dist
                detection_dist = self.rep_invs_r
                if dist < detection_dist:
                    factor = (detection_dist - dist) / detection_dist
                    magnitude = factor
                else:
                    magnitude = 0
                rep_dir = push_dir * magnitude
        return rep_dir
        
    
    def sigmoid(self, x):
        #sigmoid function
        return 1 / (1 + np.exp(-x))
    
    def form_vortex_field_circle(self, unit: Prime_unit, obstacle=None, mock_position=None):
        self.rep_in_form = 20.0
        if mock_position is not None:
            my_pos = mock_position
        else:
            my_pos = self.position
        unit_pos = unit.position
        unit_vel = unit.curr_speed
        #the center of the vortex directly at prime
        rel_unit_pos = my_pos - unit_pos
        #radius is combination of two according to prime fly direction
        sigm = self.sigmoid(np.dot(unit_vel, rel_unit_pos))
        form_r = sigm * self.formation_r + (1 - sigm) * self.formation_r_min
        dist = np.linalg.norm(rel_unit_pos) - unit.my_rad - self.my_rad
        if dist < 1e-6:
            return np.zeros_like(my_pos)
        rho = 1 - (dist / form_r)**2
        #inside of circle
        if rho > 0:
            alpha = 20.0
        #outside of circle
        else:
            alpha = 1.0
        norm_vec = rel_unit_pos/np.linalg.norm(rel_unit_pos)
        #circle around center
        tangent_vec = np.array([-rel_unit_pos[1], rel_unit_pos[0]])
        tan_len = np.linalg.norm(tangent_vec)
        if tan_len > 1e-6 or tan_len > self.circle_tan_max:
            tangent_vec = (tangent_vec / tan_len) * self.circle_tan_max
        fdbck = alpha * rho * norm_vec
        #direction is also determined by the position of obstacle
        circle_dir = self.circle_dir
        if obstacle is not None:
            if self.get_avoidance_direction(obstacle['center'], obstacle['radius'], unit):
                circle_dir = self.circle_dir_obs
                self.rep_in_form = 5.0
        fdwrd = circle_dir * tangent_vec# * self.max_speed
        form_vel = fdbck + fdwrd
        if rho < 0:
            form_norm = np.linalg.norm(form_vel)
            if form_norm > self.max_form_speed:
                form_vel = (form_vel/form_norm)*self.max_form_speed
        return form_vel
    
    def calculate_axis_consensus(self, neighbors, center_pos):
        if not neighbors:
            return None
        #sum of all axes
        #positions and speeds from everyone
        n_pos = np.vstack([n.position for n in neighbors])
        n_speed = np.vstack([n.curr_speed for n in neighbors])
        #relative position
        n_rel_pos = n_pos - center_pos
        #cross product
        n_axes = np.cross(n_rel_pos, n_speed)
        #norms
        norms = np.linalg.norm(n_axes, axis=1)
        #filtering those too small
        valid_mask = norms > 1e-6
        valid_axes = n_axes[valid_mask]
        valid_norms = norms[valid_mask]
        #norming the axes
        normalized_axes = valid_axes / valid_norms[:, np.newaxis]
        #final count
        axes_sum = np.sum(normalized_axes, axis=0) 
        count = len(normalized_axes)
        #fallback division of zero
        if count == 0:
            return None
        #average axis
        avg_axis = axes_sum / count
        #normalize
        if np.linalg.norm(avg_axis) > 1e-6:
            return avg_axis / np.linalg.norm(avg_axis)
        return None
    
    def calculate_obstacle_avoidance_axis(self, unit, obstacle_pos, obstacle_rad):
        #calculating if prime is too close to obstacle
        vec_to_obs = obstacle_pos - unit.position
        dist = np.linalg.norm(vec_to_obs) - unit.my_rad - obstacle_rad
        #prime velocity vector
        velocity = unit.curr_speed
        speed = np.linalg.norm(unit.curr_speed)
        v = np.zeros_like(self.position)
        if speed > 1e-6:
            v = velocity/speed
        vec_to_obs_norm = np.linalg.norm(vec_to_obs)
        vto_normed = np.zeros_like(self.position)
        if vec_to_obs_norm > 1e-6:
            vto_normed = vec_to_obs/vec_to_obs_norm
        if dist > self.obs_rad or np.dot(v, vto_normed) <= -0.2:
            self.avoid_axis = None
            return None
        elif self.avoid_axis is not None:
            return self.avoid_axis
        if np.linalg.norm(velocity) < 0.1:
            velocity = np.array([0.0, 0.0, 1.0])
        #axis for avoidance
        avoidance_axis = np.cross(velocity, vec_to_obs)
        #if prime flies directly into the obstacle
        norm = np.linalg.norm(avoidance_axis)
        if norm < 1e-6:
            avoidance_axis = np.cross(velocity, np.array([0,0,1]))
            norm = np.linalg.norm(avoidance_axis)
        avoidance_axis = -avoidance_axis / norm
        self.avoid_axis = avoidance_axis
        return self.avoid_axis

    def get_axis_at_time(self, t):
        #school of fish
        time_scaled = t * 0.3
        axis = np.array([np.sin(time_scaled), np.cos(time_scaled * 0.7), np.sin(time_scaled * 1.2)])
        norm = np.linalg.norm(axis)
        if norm > 1e-6:
            return axis / norm
        return np.array([0.0, 0.0, 1.0])

    def form_vortex_field_sphere(self, unit: Prime_unit, obstacle=None, mock_position=None, close_purs=None):
        if mock_position is not None:
            my_pos = mock_position
        else:
            my_pos = self.position
        unit_pos = unit.position
        unit_vel = unit.curr_speed
        rel_unit_pos = my_pos - unit_pos
        #the radius is according to the position of pursuer w.r.t. prime's velocity vector
        sigm = self.sigmoid(np.dot(unit_vel, rel_unit_pos))
        form_r = sigm * self.formation_r + (1 - sigm) * self.formation_r_min
        #distance from prime
        dist = np.linalg.norm(rel_unit_pos) - unit.my_rad - self.my_rad
        if dist < 1e-6:
            return np.zeros_like(my_pos)
        rho = 1 - (dist / form_r)**2
        #alpha, stronger on the inside
        if rho > 0:
            alpha = 20.0
        else:
            alpha = 1.0
        #normalized vec from prime to pursuer
        normal_vec = rel_unit_pos / np.linalg.norm(rel_unit_pos)
        final_axis = None
        #observed average axis of neighbors
        observed_group_axis = self.calculate_axis_consensus(close_purs, unit.position)
        #axis to avoid obstacle
        if obstacle is not None:
             final_axis = self.calculate_obstacle_avoidance_axis(unit, obstacle['center'], obstacle['radius'])
             force_flatten = 0
        #if obstacle is not close, do school of fish
        if final_axis is None:
            self.rep_in_form = 20.0
            #my axis
            current_axis = self.get_axis_at_time(self.num_iter)
            #synchronization of clocks
            if observed_group_axis is not None:
                #current alignment
                current_alignment = np.dot(current_axis, observed_group_axis)
                # if alignment < 0.7:
                #     print(f"Alignment: {current_alignment:.4f}")
                #bad sync, searching for better
                if current_alignment < 0.8:
                    #searching through the whole period
                    search_range = np.linspace(self.num_iter - 105.0, self.num_iter + 105.0, 100)
                    #computing axes
                    time_scaled = search_range * 0.3
                    x = np.sin(time_scaled)
                    y = np.cos(time_scaled * 0.7)
                    z = np.sin(time_scaled * 1.2)
                    #matrix of axes
                    axes = np.column_stack((x, y, z))
                    #norms
                    norms = np.linalg.norm(axes, axis=1)
                    #valid norms
                    valid_mask = norms > 1e-6
                    #norming the axes
                    axes[valid_mask] = axes[valid_mask] / norms[valid_mask, np.newaxis]
                    #too small axes being this default
                    axes[~valid_mask] = np.array([0.0, 0.0, 1.0])
                    #dot product of all axes
                    scores = np.dot(axes, observed_group_axis)
                    #the best alignment
                    best_idx = np.argmax(scores)
                    best_score = scores[best_idx]
                    best_t = search_range[best_idx]
                    #best time
                    self.num_iter = best_t
                    #if the dot product is very bad, try random jump
                    if best_score < 0.0:
                         self.num_iter += np.random.uniform(10.0, 200.0)
                #fine tuning, axes are very similar
                else:
                    #smaller jump, testing if setting clocks forward or backward makes better alignment
                    delta_test = 0.3
                    axis_future = self.get_axis_at_time(self.num_iter + delta_test)
                    axis_past = self.get_axis_at_time(self.num_iter - delta_test)
                    #making dot products
                    score_future = np.dot(axis_future, observed_group_axis)
                    score_past = np.dot(axis_past, observed_group_axis)
                    #setting clocks to better result
                    sync_speed = 0.2
                    if score_future > current_alignment:
                         self.num_iter += sync_speed
                    elif score_past > current_alignment:
                         self.num_iter -= sync_speed
                #new my final axis
                final_axis = self.get_axis_at_time(self.num_iter)
                #combination of my axis with the observed one
                weight = 0.5 
                final_axis = (1 - weight) * final_axis + weight * observed_group_axis
                #norm
                norm = np.linalg.norm(final_axis)
                if norm > 1e-6:
                    final_axis /= norm
            else:
                #being alone
                final_axis = current_axis
            force_flatten = 0
        else:
            self.rep_in_form = 5.0
            #force to flatten formation so that the obstacle avoidance would work
            dist_from_plane = np.dot(rel_unit_pos, final_axis)
            force_flatten = -final_axis * dist_from_plane * 20.0
        #resulting tangent
        tangent_vec = np.cross(final_axis, rel_unit_pos)
        tan_len = np.linalg.norm(tangent_vec)
        if tan_len > 1e-6:
            tangent_vec = tangent_vec / tan_len
        #composing the forces into resulting form vector
        fdbck = alpha * rho * normal_vec
        fdwrd = self.circle_dir * tangent_vec * self.max_speed
        form_vel = fdwrd + fdbck + force_flatten
        if rho < 0:
            form_norm = np.linalg.norm(form_vel)
            if form_norm > self.max_form_speed:
                form_vel = (form_vel/form_norm)*self.max_form_speed
        return form_vel
    
    def pursue_target(self, target: list[Invader, int], purs: list[Agent], unit: Prime_unit):
        tar_speed = np.linalg.norm(target[0].curr_speed)
        my_speed = self.cruise_speed
        #if target is faster then pursuer, just pure pursue him
        if tar_speed >= my_speed or target[1] == self.purs_types['pure_pursuit']:
            target[1] = self.purs_types['pure_pursuit']
            return self.pursuit_pure_pursuit(target)
        #still too fast for encirclement, CB him
        elif tar_speed >= my_speed/1.2 or target[1] == self.purs_types['const_bear']:
            target[1] = self.purs_types['const_bear']
            return self.pursuit_constant_bearing(target)
        #if more then one is chasing him and he is further from unit, circle him
        if np.linalg.norm(unit.position - target[0].position) >= self.safe_circle_r:
            # for p in purs:
            #     if p is not self and p.target != None and p.target[0] is target[0]:
            if target[0].purs_num >= 2:
                if len(self.position) == 2:
                    return self.pursuit_circling(target)
                else:
                    return self.pursuit_sphering(target, purs, unit)
        #no one else is chasing him, catch him
        target[1] = self.purs_types['const_bear1']
        return self.pursuit_constant_bearing(target)
    
    def pursuit_circling(self, target: list[Invader, int], mock_position=None):
        if mock_position is not None:
            my_pos = mock_position
        else:
            my_pos = self.position
        target[1] = self.purs_types['circling']
        self.prime_coll_r = 10.5
        #strong repulsive force is needed
        self.rep_in_purs = 25.0
        self.prime_rep_in_purs = 20.9
        #the center of the vortex field is not shifted
        rel_pos = my_pos - (target[0].position) #+ target[0].curr_speed * self.dt * self.pred_time)
        dist = np.linalg.norm(rel_pos) - target[0].my_rad - self.my_rad
        if dist < 1e-6:
            return np.zeros_like(my_pos)
        rho = 1 - (dist / self.t_circle)**2
        #inside of circle
        if rho > 0:
            alpha = 20.0
        #outside of circle
        else:
            alpha = 1.0
        norm_vec = rel_pos/np.linalg.norm(rel_pos)
        #circling in opposite direction to defensive formation circle
        tangent_vec = np.array([-rel_pos[1], rel_pos[0]])
        tan_norm = np.linalg.norm(tangent_vec)
        if tan_norm > 1e-6:
            tangent_vec = tangent_vec / tan_norm
        fdbck = alpha * rho * norm_vec
        fdwrd = -self.circle_dir * tangent_vec
        purs_vel = fdbck + fdwrd
        purs_norm = np.linalg.norm(purs_vel)
        if purs_norm > self.cruise_speed:
            purs_vel = (purs_vel/purs_norm)*self.cruise_speed
        #normalizing it to not fly that fast, risk of collision
        vel_norm = np.linalg.norm(purs_vel)
        vel_dot = np.dot(target[0].curr_speed, self.curr_speed)
        if self.t_circle * 8.0 >= dist >= self.t_circle * 3.0 and vel_norm >= 3.0 and vel_dot < 0:
            purs_vel = purs_vel/vel_norm * 0.9
        return purs_vel
    
    def calculate_prime_avoidance_axis(self, inv, prime_pos):
        vec_to_prime = prime_pos - inv.position
        #inv velocity vector
        velocity = inv.curr_speed
        if np.linalg.norm(velocity) < 0.1:
            velocity = np.array([0.0, 0.0, 1.0])
        #axis for avoidance
        avoidance_axis = np.cross(velocity, vec_to_prime)
        #if inv flies directly into the prime
        norm = np.linalg.norm(avoidance_axis)
        if norm < 1e-6:
            avoidance_axis = np.cross(velocity, np.array([0,0,1]))
            norm = np.linalg.norm(avoidance_axis)
        avoidance_axis = -avoidance_axis / norm
        return avoidance_axis
    
    def pursuit_sphering(self, target: list[Invader, int], purs, prime):
        #other pursuers pursuing same target in visibility region
        other_purs = []
        for p in purs:
            if p is not self and p.target != None and p.target[0] is target[0]:
                other_purs.append(p)
        target[1] = self.purs_types['circling']
        #strong repulsive force is needed
        self.prime_coll_r = 10.5
        self.rep_in_purs = 25.0
        self.prime_rep_in_purs = 20.9
        my_pos = self.position
        rel_pos = my_pos - (target[0].position)
        #distance from target
        dist = np.linalg.norm(rel_pos) - target[0].my_rad - self.my_rad
        if dist < 1e-6:
            return np.zeros_like(my_pos)
        rho = 1 - (dist / self.t_circle)**2
        #alpha, stronger on the inside
        if rho > 0:
            alpha = 20.0
        else:
            alpha = 1.0
        #normalized vec from target to pursuer
        normal_vec = rel_pos / np.linalg.norm(rel_pos)
        #if prime is too close to target, push target out
        prime_dist = np.linalg.norm(prime.position - target[0].position) - target[0].my_rad - prime.my_rad
        if prime_dist < self.prime_coll_r:
            final_axis = self.calculate_prime_avoidance_axis(target[0], prime.position)
            #force to flatten formation so that the obstacle avoidance would work
            dist_from_plane = np.dot(rel_pos, final_axis)
            force_flatten = -final_axis * dist_from_plane * 20.0
        else:
            #rotation axis, cool looking
            t = self.num_iter * 0.3
            desired_axis = np.array([np.sin(t), np.cos(t * 0.7), np.sin(t * 1.2)])
            desired_axis = desired_axis / np.linalg.norm(desired_axis)
            #average axis of all neighbors
            consensus_axis = self.calculate_axis_consensus(other_purs, target[0].position)
            #weighted sum of those
            if consensus_axis is not None:
                weight = 0.8 
                final_axis = (1 - weight) * desired_axis + weight * consensus_axis
                final_axis /= np.linalg.norm(final_axis)
            else:
                final_axis = desired_axis
            force_flatten = 0.0
        #resulting tangent
        tangent_vec = np.cross(final_axis, rel_pos)
        tan_norm = np.linalg.norm(tangent_vec)
        if tan_norm > 1e-6:
            tangent_vec = tangent_vec / tan_norm
        #composing the forces into resulting form vector
        fdbck = alpha * rho * normal_vec
        fdwrd = self.circle_dir * tangent_vec
        purs_vel = fdwrd + fdbck + force_flatten
        purs_norm = np.linalg.norm(purs_vel)
        if purs_norm > self.cruise_speed:
            purs_vel = (purs_vel/purs_norm)*self.cruise_speed
        return purs_vel
        
    def pursuit_constant_bearing(self, target: list[Invader, int]):
        #target is quite fast, circling is not possible
        self.prime_coll_r = 1.5
        self.rep_in_purs = 2.0
        self.prime_rep_in_purs = 12.9
        v_tar = target[0].curr_speed
        #line of sight
        r = target[0].position - self.position
        #coeficients of quadratic equation
        a, b, c = np.dot(r, r), -2*np.dot(v_tar, r), np.dot(v_tar, v_tar) - self.cruise_speed**2
        #discriminant
        D = b**2 - 4*a*c
        CB_dir = np.zeros_like(self.position)
        #positive D
        if D >= 1e-6:
            lambda1, lambda2 = (-b + np.sqrt(D))/(2*a), (-b - np.sqrt(D))/(2*a)
            CB_dir1, CB_dir2 = v_tar - lambda1*r, v_tar - lambda2*r
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
        self.prime_coll_r = 1.5
        self.rep_in_purs = 2.0
        self.prime_rep_in_purs = 12.9
        PP_dir = target[0].position - self.position
        #norming it to the max speed
        if np.linalg.norm(PP_dir) < 1e-12:
            return np.zeros_like(PP_dir)
        PP_dir = PP_dir / np.linalg.norm(PP_dir)
        PP_dir = PP_dir * self.max_speed
        return PP_dir
    
# =============== CURRENTLY UNUSED CODE ======================

    # def repulsive_force_in_circling_pursuit(self, prime, target):
    #     #calculation rel pos of pursuer, finding out if pursuer is in between prime and center of mass
    #     vec_to_target = target.position - self.position
    #     vec_to_prime = prime.position - self.position
    #     norm_t = np.linalg.norm(vec_to_target)
    #     norm_p = np.linalg.norm(vec_to_prime)
    #     if norm_t > 1e-6 and norm_p > 1e-6:
    #         vec_a = vec_to_target / norm_t
    #         vec_b = vec_to_prime / norm_p
    #         dot_prod = np.dot(vec_a, vec_b)
    #         if dot_prod < 0.0:
    #             return True            
    #     return False

    # def rotate(self, v, angle_rad):
    #     c, s = np.cos(angle_rad), np.sin(angle_rad)
    #     R = np.array([[c, -s],
    #                 [s,  c]])
    #     return R @ v

    # def form_vortex_field_ellipse(self, unit: Prime_unit):
    #     my_pos = self.position
    #     unit_pos = unit.position
    #     unit_vel = unit.curr_speed
    #     if self.position.size == 3:
    #         my_pos = np.delete(my_pos, -1)
    #         unit_pos = np.delete(unit_pos, -1)
    #         unit_vel = np.delete(unit_vel, -1)
    #     rot_angle = np.arctan2(unit_vel[1], unit_vel[0])
    #     rel_speed = np.linalg.norm(unit_vel) #/self.max_speed
    #     #scaling axes
    #     axis_a = max(2.0*rel_speed, self.formation_r)
    #     axis_b = max(1.3*rel_speed, self.formation_r)
    #     if rel_speed <= 0.1:
    #         rel_center = np.array([0, 0])
    #     else:
    #         rel_center = np.array([-0.7*axis_a, 0])
    #     center = unit_pos - self.rotate(rel_center, rot_angle)
    #     #the center of the vortex field shifted in the current unit speed vector, because unit is moving
    #     rel_pos = self.rotate(my_pos - center, -rot_angle)
    #     #rel_norm_pos = self.rotate(my_pos - unit_pos, -rot_angle)
    #     rho = 1 - (rel_pos[0]/axis_a)**2 - (rel_pos[1]/axis_b)**2
    #     loc_norm = np.array([2*rel_pos[0]/axis_a**2, 2*rel_pos[1]/axis_b**2])
    #     norm = self.rotate(loc_norm, rot_angle)
    #     normalized = norm/np.linalg.norm(norm)
    #     #inside of circle
    #     if rho > 0:
    #         alpha = 7.0
    #     #outside of circle
    #     else:
    #         alpha = 1.0
    #     #circle around center
    #     loc_fdwrd = np.array([(-axis_a/axis_b)*rel_pos[1], (axis_b/axis_a)*rel_pos[0]])
    #     fdwrd = self.rotate(loc_fdwrd, rot_angle)
    #     fdbck = np.array([alpha*normalized[0]*rho, alpha*normalized[1]*rho])
    #     form_vel = self.circle_dir*fdwrd + fdbck
    #     #safety measure
    #     # diff_vec = my_pos - unit_pos
    #     # dist = np.linalg.norm(diff_vec)
    #     # safe_radius = 0.5
    #     # avoidance_vel = np.zeros_like(form_vel)
    #     # if dist < safe_radius and dist > 0.001:
    #     #     push_dir = diff_vec / dist
    #     #     repulsion_strength = 5.0 * (safe_radius - dist) / safe_radius 
    #     #     avoidance_vel = push_dir * repulsion_strength
    #     # final_vel = form_vel + avoidance_vel
    #     return form_vel
    
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
    
    # def pursuit_augmented_PN(self, target, dt=0.1, N=3.0, debug=False):
    #     if self.ema_acc.shape != target.curr_acc.shape:
    #         self.ema_acc = np.zeros_like(target.curr_acc)
    #     r = target.position - self.position
    #     r_norm = np.linalg.norm(r)
    #     if r_norm < 1e-4:
    #         return self.curr_speed.copy()
    #     r_hat = r / r_norm
    #     v_rel = target.curr_speed - self.curr_speed
    #     V_c = -np.dot(v_rel, r_hat)
    #     LOS_der = (v_rel - np.dot(v_rel, r_hat) * r_hat) / r_norm
    #     alpha = 0.2
    #     self.ema_acc = alpha * target.curr_acc + (1.0 - alpha) * self.ema_acc
    #     a_T = self.ema_acc
    #     a_T_perp = a_T - np.dot(a_T, r_hat) * r_hat
    #     closing_speed_for_gain = max(V_c, 1.0)
    #     pn_term = N * closing_speed_for_gain * LOS_der
    #     apn_term = (N / 2) * a_T_perp
    #     acc_cmd = pn_term + apn_term
    #     # DEBUG
    #     if debug:
    #         print(f"PN_force: {np.linalg.norm(pn_term):.2f}, APN_force: {np.linalg.norm(apn_term):.2f}")
    #     if np.linalg.norm(acc_cmd) > self.max_acc:
    #         acc_cmd = acc_cmd / np.linalg.norm(acc_cmd) * self.max_acc
    #     v_new = self.curr_speed + acc_cmd * dt
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