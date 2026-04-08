import numpy as np
from agent import Agent
from invader import Invader
from prime_unit import Prime_unit
from pursuer_states import States
import copy
from stable_baselines3 import PPO

class Pursuer(Agent):
    def __init__(self, position, max_speed, max_acc, max_omega, my_rad, purs_num, purs_vis, dt, pursue_model: PPO | None = None, def_model: PPO | None = None):
        super().__init__(position, max_speed, max_acc, max_omega, dt, my_rad, num_iter=purs_num)
        self.is_rl_controlled = False
        rnd_num = np.random.randint(0, 2)
        #updating the brain
        if isinstance(pursue_model, tuple):
            if rnd_num == 0:
                self.pursue_model = pursue_model[0]
            else:
                self.pursue_model = pursue_model[1]
        else:
            self.pursue_model = pursue_model
        self.def_model = None
        if def_model is not None:
            self.is_rl_controlled = True
            self.def_model = def_model
        #visibility range
        self.vis_range = purs_vis
        #repulsive forces
        self.purs = 1.2
        self.form = 0.9
        self.rep_in_form = 2.3
        self.rep_in_purs = 0.7
        self.rep_obs = 1.9
        self.rep_gr = 1.9
        self.rep_invs = 2.3
        self.prime_rep_in_purs = 2.5
        #collision radiuses
        self.prime_coll_r = 10.5
        self.collision_r = min(purs_vis, 2.0)
        self.coll_gr = 2.0
        #formation radiuses
        #self.formation_r = 0.9
        #self.formation_r_min = 0.4
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
        # self.obs_rad = 8.0
        # self.rep_obs_r = 8.0
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
        if self.pos_length == 3:
            self.MAX_PURSUERS = 3
        else:
            self.MAX_PURSUERS = 4
        #obstacles centers and radiuses
        self.obs_centers = None
        self.obs_radii = None
        self.coll_obs = 5.0
        #max speed in formation
        self.max_form_speed = 3.0
        #gauss noise
        self.base_pos = 0.01
        self.k_pos = 0.02
        self.base_vel = 0.1
        self.k_vel = 0.01
        self.base_rad = 0.02
        self.k_rad = 0.02
        self.base_acc = 0.02
        self.k_acc = 0.02
        self.base_ang = 0.05
        self.k_ang = 0.005
        self.new_acc_vector = None
        self.my_clock = -1
        self.new_acc = np.zeros_like(self.position)
        #cooldown of RL attack decision
        self.cooldown = 0
        self.tried_invalid_attack = False
        self.override_active = False
        
    def herding(self, acc_vector):
        #getting the vector, nothing else
        if self.target is not None:
            self.target["purs_type"] = self.purs_types["circling"]
        self.new_acc_vector = acc_vector
        
    def pursue_herding(self):
        observation = self.get_observation_herding()
        action, _ = self.pursue_model.predict(observation, deterministic=True)
        raw_act = np.array(action, dtype=np.float32)
        #cliping
        norm = np.linalg.norm(raw_act)
        if norm > 1.0:
            raw_act = raw_act / norm
        target_acc = raw_act * self.max_acc
        self.new_acc = target_acc
        return
    
    def defense(self, targets):
        observation = self.get_observation()
        #observation = self.get_observation_restrictive()
        action, _ = self.def_model.predict(observation, deterministic=True)
        #vis_inv_0 = self.get_closest_invaders(targets, 2)
        vis_inv_0 = self.current_tactical_invaders
        #self.set_rl_action_restrictive(action, vis_inv_0)
        self.set_rl_action(action, vis_inv_0)
        
    def pursue(self, targets: list[Invader], prime_vel, prime_rad, prime_pos, all_purs_tars, precalc_data, not_testing=False, no_target=False):
        if not_testing:
            if self.my_clock % 20 == 0 and self.is_rl_controlled and self.def_model is not None:
                self.defense(targets)
            self.my_clock += 1
            if self.my_clock % 5 != 0:
                return self.new_acc
        #precalculated data, faster this way
        self.prime_rad = prime_rad
        all_inv_pos, all_inv_vel, self.all_inv_purs_num, all_inv_rads, all_inv_ang_vel, all_inv_acc, all_purs_pos, all_purs_vel, all_purs_rads, all_purs_ang_vel, all_purs_acc, self.my_index, obs_centers, obs_radii = precalc_data
        #Pursuers
        self.all_purs_pos, self.all_purs_vel, self.all_purs_rads, self.all_purs_acc, self.all_purs_ang_vel, self.all_purs_tars = self.get_visible_neighbors_data(
            all_purs_pos.copy(), all_purs_vel.copy(), all_purs_rads.copy(), all_purs_tars, all_purs_acc.copy(), all_purs_ang_vel.copy(), gaussian=True)    
        #Invaders
        self.all_inv_pos, self.all_inv_vel, self.all_inv_rads, self.all_inv_acc, self.all_inv_ang_vel, inv_mask = self.filter_visible_objects(
            all_inv_pos.copy(), all_inv_vel.copy(), all_inv_rads.copy(), accs=all_inv_acc.copy(), ang_vels=all_inv_ang_vel.copy(), max_range=30.0, apply_noise=True)
        self.all_inv_purs_num = self.all_inv_purs_num[inv_mask]
        targets = [obj for obj, m in zip(targets, inv_mask) if m]
        self.targets = targets
        #Obstacles
        if obs_centers is not None and len(obs_centers) > 0:
            self.obs_centers, _, self.obs_radii, _, _, _ = self.filter_visible_objects(
                obs_centers.copy(), None, obs_radii.copy(), max_range=30.0, apply_noise=True, obstacles=True)
        else:
            self.obs_centers, self.obs_radii = None, None
        #The rest
        self.copy_data(prime_vel, prime_pos)
        if self.target:
            self.update_target()
        self.gauss_noise()
        #closest_inv = self.get_closest_invaders(targets, max_count=1)
        # if len(closest_inv) != 0 and 28 <= np.linalg.norm(closest_inv[0].position - self.position) <= 30:
        #     if self.is_rl_controlled and self.def_model is not None:
        #         self.defense(targets)
        #init directions
        tar_vel = np.zeros_like(self.position)
        form_vel = np.zeros_like(self.position)
        #if crashed, dont move, you are supposed to be dead
        if self.crashed:
            return tar_vel
        if self.new_acc_vector is not None:
            return self.new_acc_vector
        #previous target captured, back to formation
        if self.target != None and self.target["target"].crashed == True:
            self.target = None
            self.cooldown = 0
            self.state = States.FORM
        #pursuer having no target, finding target, if found, pursue it
        if self.target == None and not self.is_rl_controlled and not no_target:
            if self.strategy_capture_cone(targets, self.num_iter, cooldown=self.capture_cooldown*self.dt):
                if self.pursue_model is not None:
                    self.pursue_herding()
                    return self.new_acc
                else:
                    tar_vel = self.pursue_target(self.target)
            elif self.strategy_target_close(targets):
                if self.pursue_model is not None:
                    self.pursue_herding()
                    return self.new_acc
                else:
                    tar_vel = self.pursue_target(self.target)
        #pursuer having target -> pursue it
        elif self.target != None and self.target["target"].crashed == False and not self.is_rl_controlled:
            if not (self.target["purs_type"] != self.purs_types["circling"] and np.linalg.norm(self.position - self.prime_pos) > self.capture_max):
                if self.pursue_model is not None:
                    self.pursue_herding()
                    return self.new_acc
                else:
                    tar_vel = self.pursue_target(self.target)
        elif self.target != None and self.target["target"].crashed == False and self.is_rl_controlled:
            tar_vel = self.pursue_rl_target(self.target)
            if self.target["purs_type"] == self.purs_types["circling"]:
                return self.new_acc
        #if target dir is zero, pursuer has no target -> keep the formation
        if np.array_equal(tar_vel, form_vel):
            self.target = None
            self.state = States.FORM
            if self.pos_length == 2:
                form_vel = self.form_vortex_field_circle()
            else:
                form_vel = self.form_vortex_field_sphere()
        #repulsive dirs to avoid collision
        rep_vel = self.repulsive_force_purs(self.collision_r)
        #repulsive dirs to avoid collision with obstacle
        obs_vel = self.repulsive_force_obs(self.coll_obs)
        #repulsive force to avoid ground
        ground_vel = self.repulsive_force_ground(self.coll_gr)
        #returning sum of those
        if not np.array_equal(form_vel, np.zeros_like(form_vel)):
            #pushing whole formation away from invaders and obstacle
            #invs_rep_vel = self.repulsive_inv_force(targets)
            sum_vel = self.rep_in_form*rep_vel + self.form*form_vel + self.rep_obs*obs_vel + self.rep_gr * ground_vel #+ self.rep_invs*invs_rep_vel
        else:
            prime_rep_vel = self.repulsive_force_prime(self.prime_coll_r)
            sum_vel = self.purs*tar_vel + self.rep_in_purs*rep_vel + self.prime_rep_in_purs*prime_rep_vel + self.rep_obs*obs_vel + self.rep_gr * ground_vel
        #norming speed to the possible limit
        sum_norm = np.linalg.norm(sum_vel)
        if sum_norm > self.cruise_speed:
            sum_vel = (sum_vel/sum_norm) * self.cruise_speed
        #converting to acceleration
        self.new_acc = self.KP * (sum_vel - self.curr_speed) - self.KD * self.curr_speed
        return self.new_acc
    
    def get_closest_invaders(self, all_invaders, max_count=2):
        if not all_invaders:
            return []
        #distances
        dists = [np.linalg.norm(inv.position - self.position) for inv in all_invaders]
        #sorted distances
        sorted_indices = np.argsort(dists)
        #closest invaders
        return [all_invaders[i] for i in sorted_indices[:max_count]]
    #MODEL B
    def observation_herding(self):
        #observation for herding NN, everything normalized
        MAX_COORD = 30.0       #world
        MAX_DIST = 40.0        #max possible distance
        MAX_SPEED = 8.5        #max speed
        MAX_DENSITY = 6.0     #max pursuer density
        MAX_RADIUS = 5.0      #max obstacle radius
        MAX_DRONE_RAD = 0.7
        FAR_AWAY = 3.0  #far away, bigger number
        MAX_ACC = 8.0
        #MY STATE
        my_obs = np.concatenate([self.curr_speed / MAX_SPEED, self.curr_acc / MAX_ACC, [self.position[2] / MAX_COORD], 
                                 [self.my_rad / MAX_DRONE_RAD], [self.cruise_speed / MAX_SPEED], [self.max_acc / MAX_ACC]]) 
        #PRIME STATE
        prime_rad_obs = np.array([self.prime_rad / MAX_DRONE_RAD], dtype=np.float32)
        prime_rel_vel = self.prime_vel - self.curr_speed
        prime_pos = (self.prime_pos - self.position) / MAX_DIST
        #dist surface - surface
        raw_prime_dist = np.linalg.norm(self.prime_pos - self.position)
        surf_prime_dist = max(0.0, raw_prime_dist - self.prime_rad - self.my_rad)
        #normalizing for neural net
        prime_purs_dist = surf_prime_dist / MAX_DIST
        prime_obs = np.concatenate([prime_pos, prime_rel_vel / (MAX_SPEED * 2.0),prime_rad_obs,
                                    [prime_purs_dist]])
        #PURSUER FILTER
        other_purs_pos = []
        other_purs_vel = []
        other_purs_rads = []
        for i, tar in enumerate(self.all_purs_tars):
            #those targeting same invader
            if tar is not None and tar is self.target["target"]:
                other_purs_pos.append(self.all_purs_pos[i])
                other_purs_vel.append(self.all_purs_vel[i])
                other_purs_rads.append(self.all_purs_rads[i])
        new_purs_pos = np.array(other_purs_pos)
        new_purs_vel = np.array(other_purs_vel)
        new_purs_rads = np.array(other_purs_rads)
        #PURSUER STATE
        pursuers_obs = np.full(32, FAR_AWAY, dtype=np.float32)
        #default settings
        for i in range(4):
            start = i * 8
            pursuers_obs[start+3 : start+6] = 0.0  #rel velocity
            pursuers_obs[start+6] = 0.0            #radius
        density = 0
        if len(new_purs_pos) > 0:
            density = len(new_purs_pos)
            #relative normalized positions
            norm_rel_positions = (new_purs_pos - self.position) / self.vis_range
            #sorting pursuers according to the distance
            norm_dists = np.linalg.norm(norm_rel_positions, axis=1)
            closest_indices = np.argsort(norm_dists)[:4]
            #iterating from closest indices
            for i, idx in enumerate(closest_indices):
                start = i * 8   
                #rel position
                pursuers_obs[start : start+3] = norm_rel_positions[idx]    
                #rel velocity
                rel_vel = new_purs_vel[idx] - self.curr_speed
                pursuers_obs[start+3 : start+6] = rel_vel / (MAX_SPEED * 2.0)    
                #radius
                pursuers_obs[start+6] = new_purs_rads[idx] / MAX_DRONE_RAD    
                #dist surface - surface
                raw_mate_to_me_dist = np.linalg.norm(new_purs_pos[idx] - self.position)
                surf_mate_to_me_dist = max(0.0, raw_mate_to_me_dist - new_purs_rads[idx] - self.my_rad)
                pursuers_obs[start+7] = surf_mate_to_me_dist / self.vis_range
        #DENSITY
        density_obs = np.array([density / MAX_DENSITY], dtype=np.float32)
        #PURSUER IN FORMATION STATE (not used)
        pursuers_form = np.full(32, FAR_AWAY, dtype=np.float32)
        #default settings
        for i in range(4):
            start = i * 8
            pursuers_form[start+3 : start+6] = 0.0  
            pursuers_form[start+6] = 0.0            
        #LOS STATE
        attack_vector = self.target["tar_pos"] - self.prime_pos
        attack_dist = np.linalg.norm(attack_vector)
        if attack_dist > 0.1:
            prime_to_me = self.position - self.prime_pos
            #projection
            projection = np.dot(prime_to_me, attack_vector) / (attack_dist**2)
            #dist to line
            cross_prod = np.cross(attack_vector, prime_to_me)
            raw_dist_to_line = np.linalg.norm(cross_prod) / attack_dist
            #vector to the line
            closest_point_on_line = self.prime_pos + (projection * attack_vector)
            vector_to_line = closest_point_on_line - self.position    
        else:
            #fallback
            projection = 0.0
            raw_dist_to_line = MAX_DIST
            vector_to_line = np.zeros(3)
        #normalizing
        los_obs = np.concatenate([[projection], [raw_dist_to_line / MAX_DIST], vector_to_line / MAX_DIST]).astype(np.float32)
        #using empty pursuers in formation block
        pursuers_form[-5:] = los_obs
        #INVADER STATE
        inv_rel_pos = (self.target["tar_pos"] - self.position) / MAX_DIST
        inv_rel_vel = self.target["tar_vel"] - self.curr_speed
        inv_to_prime_vec = (self.prime_pos - self.target["tar_pos"]) / MAX_DIST
        #dist to prime surface - surface
        raw_inv_prime_dist = np.linalg.norm(self.prime_pos - self.target["tar_pos"])
        surf_inv_prime_dist = max(0.0, raw_inv_prime_dist - self.target["tar_rad"] - self.prime_rad)
        inv_to_prime_dist = np.array([surf_inv_prime_dist / MAX_DIST], dtype=np.float32)
        #dist to me surface - surface
        raw_inv_purs_dist = np.linalg.norm(self.target["tar_pos"] - self.position)
        surf_inv_purs_dist = max(0.0, raw_inv_purs_dist - self.target["tar_rad"] - self.my_rad)
        inv_purs_dist = np.array([surf_inv_purs_dist / MAX_DIST], dtype=np.float32)
        #radius
        inv_rad_obs = np.array([self.target["tar_rad"] / MAX_DRONE_RAD], dtype=np.float32)
        invaders_obs = np.concatenate([inv_rel_pos, inv_rel_vel / (MAX_SPEED * 2.0), inv_to_prime_vec, inv_to_prime_dist,
                       inv_rad_obs, inv_purs_dist])
        #OBSTACLE STATE
        obstacles_obs = np.full(20, FAR_AWAY, dtype=np.float32)
        #default radius
        obstacles_obs[3] = 0.0 
        obstacles_obs[8] = 0.0 
        obstacles_obs[13] = 0.0 
        obstacles_obs[18] = 0.0 
        #centers and radii of obstacles
        obs_centers = self.obs_centers
        obs_radii = self.obs_radii
        if obs_centers is not None and len(obs_centers) > 0:
            #dist to me surface - surface
            center_dists = np.linalg.norm(obs_centers - self.position, axis=1)
            edge_dists = center_dists - obs_radii - self.my_rad
            #filtering those in visibility range
            visible_mask = edge_dists <= self.vis_range
            visible_indices = np.where(visible_mask)[0]
            if len(visible_indices) > 0:
                #only the four closest ones
                visible_edge_dists = edge_dists[visible_indices]
                sorted_local_indices = np.argsort(visible_edge_dists)[:4]
                closest_obs_indices = visible_indices[sorted_local_indices]
                #write to the obs space
                for i, idx in enumerate(closest_obs_indices):
                    start = i * 5
                    #rel position
                    obs_pos = (obs_centers[idx] - self.position) / self.vis_range
                    obstacles_obs[start : start+3] = obs_pos
                    #radius
                    obstacles_obs[start+3] = obs_radii[idx] / MAX_RADIUS
                    #distance to obstacle
                    obstacles_obs[start+4] = np.linalg.norm(obs_pos)
        #final vector
        final_obs = np.concatenate([
            my_obs,         # 10
            prime_obs,      # 8
            density_obs,    # 1
            pursuers_obs,   # 32
            pursuers_form,  # 32
            invaders_obs,   # 12
            obstacles_obs   # 20
        ]).astype(np.float32)
        return final_obs
    #MODEL A
    def observation_herding(self):
        #observation for herding NN, everything normalized
        MAX_COORD = 30.0       #world
        MAX_DIST = 40.0        #max possible distance
        MAX_SPEED = 8.5        #max speed
        MAX_DENSITY = 6.0     #max pursuer density
        MAX_RADIUS = 5.0      #max obstacle radius
        MAX_DRONE_RAD = 0.7
        FAR_AWAY = 3.0  #far away, bigger number
        MAX_ACC = 8.0
        #MY STATE
        my_obs = np.concatenate([self.curr_speed / MAX_SPEED, self.curr_acc / MAX_ACC, [self.position[2] / MAX_COORD], 
                                 [self.my_rad / MAX_DRONE_RAD], [self.cruise_speed / MAX_SPEED], [self.max_acc / MAX_ACC]]) 
        #PRIME STATE
        prime_rad_obs = np.array([self.prime_rad / MAX_DRONE_RAD], dtype=np.float32)
        prime_rel_vel = self.prime_vel - self.curr_speed
        prime_pos = (self.prime_pos - self.position) / MAX_DIST
        prime_purs_dist = np.linalg.norm(prime_pos)
        prime_obs = np.concatenate([prime_pos, prime_rel_vel / (MAX_SPEED * 2.0), prime_rad_obs, [prime_purs_dist]])
        #PURSUERS STATE
        pursuers_obs = np.full(24, FAR_AWAY, dtype=np.float32)
        #default
        for i in range(3):
            start = i * 8
            pursuers_obs[start+3 : start+6] = 0.0  #rel velocity
            pursuers_obs[start+6] = 0.0            #radius
        density = 0
        #filtering pursuers
        other_purs_pos = []
        other_purs_vel = []
        other_purs_rads = []
        for i, tar in enumerate(self.all_purs_tars):
            #those targeting the same invader
            if tar is not None and tar is self.target["target"]:
                other_purs_pos.append(self.all_purs_pos[i])
                other_purs_vel.append(self.all_purs_vel[i])
                other_purs_rads.append(self.all_purs_rads[i])
        new_purs_pos = np.array(other_purs_pos)
        new_purs_vel = np.array(other_purs_vel)
        new_purs_rads = np.array(other_purs_rads)
        if len(new_purs_pos) > 0:
            #density of pursuer around me
            density = len(new_purs_pos)
            #normalized distance and filtering
            norm_rel_positions = (new_purs_pos - self.position) / self.vis_range
            norm_dists = np.linalg.norm(norm_rel_positions, axis=1)
            closest_indices = np.argsort(norm_dists)[:3]
            #iterating from closest indices
            for i, idx in enumerate(closest_indices):
                start = i * 8  
                #rel position
                pursuers_obs[start : start+3] = norm_rel_positions[idx]    
                #rel velocity
                rel_vel = new_purs_vel[idx] - self.curr_speed
                pursuers_obs[start+3 : start+6] = rel_vel / (MAX_SPEED * 2.0)    
                #radius
                pursuers_obs[start+6] = new_purs_rads[idx] / MAX_DRONE_RAD    
                #distance to me center - center
                pursuers_obs[start+7] = norm_dists[idx]
        #DENSITY STATE
        density_obs = np.array([density / MAX_DENSITY], dtype=np.float32)
        #INVADER STATE
        #rel pos
        inv_rel_pos = (self.target["tar_pos"] - self.position) / MAX_DIST
        #rel velocity
        inv_rel_vel = self.target["tar_vel"] - self.curr_speed
        #inv to prime vector
        inv_to_prime_vec = (self.prime_pos - self.target["tar_pos"]) / MAX_DIST
        #distances
        inv_to_prime_dist = np.array([np.linalg.norm(inv_to_prime_vec)], dtype=np.float32)
        inv_purs_dist = np.array([np.linalg.norm(inv_rel_pos)], dtype=np.float32)
        #radius
        inv_rad_obs = np.array([self.target["tar_rad"] / MAX_DRONE_RAD], dtype=np.float32)
        invaders_obs = np.concatenate([inv_rel_pos, inv_rel_vel / (MAX_SPEED * 2.0), inv_to_prime_vec, inv_to_prime_dist,
                                       inv_rad_obs, inv_purs_dist])
        #OBSTACLE STATE
        obstacles_obs = np.full(10, FAR_AWAY, dtype=np.float32)
        #default
        obstacles_obs[3] = 0.0 
        obstacles_obs[8] = 0.0 
        obs_centers = self.obs_centers
        obs_radii = self.obs_radii
        if obs_centers is not None and len(obs_centers) > 0:
            #dist to suraface
            center_dists = np.linalg.norm(obs_centers - self.position, axis=1)
            edge_dists = center_dists - obs_radii - self.my_rad
            #filtering those in visibility range
            visible_mask = edge_dists <= self.vis_range
            visible_indices = np.where(visible_mask)[0]
            if len(visible_indices) > 0:
                #filtering two closest
                visible_edge_dists = edge_dists[visible_indices]
                sorted_local_indices = np.argsort(visible_edge_dists)[:2]
                closest_obs_indices = visible_indices[sorted_local_indices]
                #write to obs space
                for i, idx in enumerate(closest_obs_indices):
                    start = i * 5
                    #rel pos
                    obs_pos = (obs_centers[idx] - self.position) / self.vis_range
                    obstacles_obs[start : start+3] = obs_pos
                    #radius
                    obstacles_obs[start+3] = obs_radii[idx] / MAX_RADIUS
                    #distance
                    obstacles_obs[start+4] = np.linalg.norm(obs_pos)
        #final vector
        final_obs = np.concatenate([
            my_obs,         # 10
            prime_obs,      # 8
            density_obs,    # 1
            pursuers_obs,   # 24
            invaders_obs,   # 12
            obstacles_obs   # 10
        ]).astype(np.float32)
        return final_obs
    #MODEL C
    def get_observation_herding(self):
        #observation for herding NN, everything normalized
        MAX_COORD = 30.0       #world
        MAX_DIST = 40.0        #max possible distance
        MAX_SPEED = 8.5        #max speed
        MAX_DENSITY = 6.0     #max pursuer density
        MAX_RADIUS = 5.0      #max obstacle radius
        MAX_DRONE_RAD = 0.7
        FAR_AWAY = 3.0  #far away, bigger number
        MAX_ACC = 8.0
        MAX_ANG_VEL = 2.0
        #MY STATE
        my_speed_mag = np.linalg.norm(self.curr_speed)
        my_acc_mag = np.linalg.norm(self.curr_acc)
        my_obs = np.concatenate([
            self.curr_speed / MAX_SPEED,      #velocity
            self.curr_acc / MAX_ACC,          #acceleration
            [my_speed_mag / MAX_SPEED],       #speed norm
            [my_acc_mag / MAX_ACC],           #acc norm
            [self.position[2] / MAX_COORD],   #height
            [self.my_rad / MAX_DRONE_RAD],    #radius
            [self.cruise_speed / MAX_SPEED],  #cruise velocity (maximum)
            [self.max_acc / MAX_ACC],         #acceleration maximum
            [self.curr_omega / MAX_ANG_VEL],  #angular speed
            [self.max_omega / MAX_ANG_VEL]    #angular maximum
        ]) 
        #PRIME STATE
        #radius
        prime_rad_obs = np.array([self.prime_rad / MAX_DRONE_RAD], dtype=np.float32)
        #rel velocity
        prime_rel_vel = self.prime_vel - self.curr_speed
        prime_pos = (self.prime_pos - self.position) / MAX_DIST
        #dist to me surface - surface
        raw_prime_dist = np.linalg.norm(self.prime_pos - self.position)
        surf_prime_dist = max(0.0, raw_prime_dist - self.prime_rad - self.my_rad)
        #normalizing
        prime_purs_dist = surf_prime_dist / MAX_DIST
        prime_obs = np.concatenate([
            prime_pos, 
            prime_rel_vel / (MAX_SPEED * 2.0),
            prime_rad_obs,
            [prime_purs_dist]
        ])
        #PURSUER FILTERING
        other_purs_pos = []
        other_purs_vel = []
        other_purs_rads = []
        other_purs_acc = []
        other_purs_ang_vel = []
        for i, tar in enumerate(self.all_purs_tars):
            #those targeting the same invader
            if tar is not None and tar is self.target["target"]:
                other_purs_pos.append(self.all_purs_pos[i])
                other_purs_vel.append(self.all_purs_vel[i])
                other_purs_rads.append(self.all_purs_rads[i])
                other_purs_acc.append(self.all_purs_acc[i])
                other_purs_ang_vel.append(self.all_purs_ang_vel[i])        
        new_purs_pos = np.array(other_purs_pos)
        new_purs_vel = np.array(other_purs_vel)
        new_purs_rads = np.array(other_purs_rads)
        new_purs_acc = np.array(other_purs_acc)
        new_purs_ang_vel = np.array(other_purs_ang_vel)
        #invader pos
        invader_pos = self.target["tar_pos"]
        #PURSUERS STATE
        pursuers_obs = np.full(45, FAR_AWAY, dtype=np.float32)
        #default
        for i in range(3):
            start = i * 15
            pursuers_obs[start+3 : start+6] = 0.0  #rel velocity
            pursuers_obs[start+6] = 0.0            #radius
            pursuers_obs[start+9 : start+12] = 0.0 #rel acc
            pursuers_obs[start+12] = 0.0           #angular speed
        density = 0
        if len(new_purs_pos) > 0:
            density = len(new_purs_pos)
            #normalized dist to me
            norm_rel_positions = (new_purs_pos - self.position) / self.vis_range
            norm_dists = np.linalg.norm(norm_rel_positions, axis=1)
            #three closest
            closest_indices = np.argsort(norm_dists)[:3]
            for i, idx in enumerate(closest_indices):
                start = i * 15
                #rel pos
                pursuers_obs[start : start+3] = norm_rel_positions[idx]    
                #rel velocity
                rel_vel = new_purs_vel[idx] - self.curr_speed
                pursuers_obs[start+3 : start+6] = rel_vel / (MAX_SPEED * 2.0)    
                #radius
                pursuers_obs[start+6] = new_purs_rads[idx] / MAX_DRONE_RAD    
                #dist to me surface - surface
                raw_mate_to_me_dist = np.linalg.norm(new_purs_pos[idx] - self.position)
                surf_mate_to_me_dist = max(0.0, raw_mate_to_me_dist - new_purs_rads[idx] - self.my_rad)
                pursuers_obs[start+7] = surf_mate_to_me_dist / self.vis_range
                #dist to invader surface - surface
                raw_mate_to_inv_dist = np.linalg.norm(new_purs_pos[idx] - invader_pos)
                surf_mate_to_inv_dist = max(0.0, raw_mate_to_inv_dist - new_purs_rads[idx] - self.target["tar_rad"])
                pursuers_obs[start+8] = surf_mate_to_inv_dist / MAX_DIST
                #rel acceleration
                rel_acc = new_purs_acc[idx] - self.curr_acc
                pursuers_obs[start+9 : start+12] = rel_acc / (MAX_ACC * 2.0)
                #angular speed
                pursuers_obs[start+12] = new_purs_ang_vel[idx] / MAX_ANG_VEL
                #norm of rel speed
                mate_speed = np.linalg.norm(new_purs_vel[idx])
                pursuers_obs[start+13] = mate_speed / MAX_SPEED
                #norm of rel acc
                mate_acc_mag = np.linalg.norm(new_purs_acc[idx])
                pursuers_obs[start+14] = mate_acc_mag / MAX_ACC
        #DENSITY STATE
        density_obs = np.array([density / MAX_DENSITY], dtype=np.float32)
        #INVADER STATE
        #rel pos
        inv_rel_pos = (self.target["tar_pos"] - self.position) / MAX_DIST
        #rel velocity
        inv_rel_vel = (self.target["tar_vel"] - self.curr_speed) / (MAX_SPEED * 2.0)
        #rel acc
        inv_rel_acc = (self.target["tar_acc"] - self.curr_acc) / (MAX_ACC * 2.0)
        #inv to prime vector
        inv_to_prime_vec = (self.prime_pos - self.target["tar_pos"]) / MAX_DIST
        #norm of speed
        inv_speed_mag = np.array([np.linalg.norm(self.target["tar_vel"]) / MAX_SPEED], dtype=np.float32)
        #norm of acc
        inv_acc_mag = np.array([np.linalg.norm(self.target["tar_acc"]) / MAX_ACC], dtype=np.float32)
        #angular speed
        inv_ang_vel = np.array([self.target["tar_ang"] / MAX_ANG_VEL], dtype=np.float32)
        #dist to prime surface - surface
        raw_inv_prime_dist = np.linalg.norm(self.prime_pos - self.target["tar_pos"])
        surf_inv_prime_dist = max(0.0, raw_inv_prime_dist - self.target["tar_rad"] - self.prime_rad)
        inv_to_prime_dist = np.array([surf_inv_prime_dist / MAX_DIST], dtype=np.float32)
        #dist to me surface - surface
        raw_inv_purs_dist = np.linalg.norm(self.target["tar_pos"] - self.position)
        surf_inv_purs_dist = max(0.0, raw_inv_purs_dist - self.target["tar_rad"] - self.my_rad)
        inv_purs_dist = np.array([surf_inv_purs_dist / MAX_DIST], dtype=np.float32)
        #radius
        inv_rad_obs = np.array([self.target["tar_rad"] / MAX_DRONE_RAD], dtype=np.float32)
        #connecting
        invaders_obs = np.concatenate([inv_rel_pos, inv_rel_vel, inv_rel_acc, inv_speed_mag, inv_acc_mag,
                                       inv_ang_vel, inv_to_prime_vec, inv_to_prime_dist, inv_rad_obs, inv_purs_dist])
        #OBSTACLE STATE
        obstacles_obs = np.full(20, FAR_AWAY, dtype=np.float32)
        #default radii
        obstacles_obs[3] = 0.0 
        obstacles_obs[8] = 0.0 
        obstacles_obs[13] = 0.0 
        obstacles_obs[18] = 0.0 
        #obstacle centers and radii
        obs_centers = self.obs_centers
        obs_radii = self.obs_radii
        if obs_centers is not None and len(obs_centers) > 0:
            #dist to centers
            center_dists = np.linalg.norm(obs_centers - self.position, axis=1)
            #dist to edges
            edge_dists = center_dists - obs_radii - self.my_rad
            #only those in sight
            visible_mask = edge_dists <= 5.0
            visible_indices = np.where(visible_mask)[0]
            if len(visible_indices) > 0:
                #only visible distances
                visible_edge_dists = edge_dists[visible_indices]
                #sorting by distance
                sorted_local_indices = np.argsort(visible_edge_dists)[:4]
                closest_obs_indices = visible_indices[sorted_local_indices]
                #writing to the obs space
                for i, idx in enumerate(closest_obs_indices):
                    start = i * 5
                    #rel position
                    obs_pos = (obs_centers[idx] - self.position) / 5.0
                    obstacles_obs[start : start+3] = obs_pos
                    #radius
                    obstacles_obs[start+3] = obs_radii[idx] / MAX_RADIUS
                    #distance
                    obstacles_obs[start+4] = np.linalg.norm(obs_pos)
        #LOS STATE
        attack_vector = self.target["tar_pos"] - self.prime_pos
        attack_dist = np.linalg.norm(attack_vector)
        if attack_dist > 0.1:
            prime_to_me = self.position - self.prime_pos
            #projection
            projection = np.dot(prime_to_me, attack_vector) / (attack_dist**2)
            projection = np.clip(projection, -3.0, 3.0)
            #dist to line
            cross_prod = np.cross(attack_vector, prime_to_me)
            raw_dist_to_line = np.linalg.norm(cross_prod) / attack_dist
            #vector to line
            closest_point_on_line = self.prime_pos + (projection * attack_vector)
            vector_to_line = closest_point_on_line - self.position    
        else:
            #fallback
            projection = 0.0
            raw_dist_to_line = MAX_DIST
            vector_to_line = np.zeros(3)
        #connecting
        los_obs = np.concatenate([[projection], [raw_dist_to_line / MAX_DIST], vector_to_line / MAX_DIST]).astype(np.float32)
        #final vector
        final_obs = np.concatenate([
            my_obs,         # 14
            prime_obs,      # 8
            density_obs,    # 1
            pursuers_obs,   # 45
            invaders_obs,   # 18
            obstacles_obs,  # 20
            los_obs,        # 5
        ]).astype(np.float32)
        return final_obs
    
    def get_observation(self):
        #observation for herding NN, everything normalized
        MAX_COORD = 30.0       #world
        MAX_DIST = 40.0        #max possible distance
        MAX_SPEED = 8.5        #max speed
        MAX_DENSITY = 15.0     #max pursuer density
        OTHER_PURS = 5.0       #max pursuer density
        MAX_RADIUS = 5.0       #max obstacle radius
        MAX_DRONE_RAD = 0.7
        FAR_AWAY = 3.0         #far away, bigger number
        MAX_ACC = 8.0
        MAX_ANG_VEL = 2.0
        MAX_COOLDOWN = 5.0   
        #STATE STATE
        #flag is attacking
        is_attacking = 1.0 if self.state == States.PURSUE else 0.0
        #cooldown
        cooldown_obs = self.cooldown / MAX_COOLDOWN
        #tactic currently used
        tactic_obs = [0.0, 0.0, 0.0]
        if self.state == States.PURSUE and self.target is not None:
            actual_strat = self.target["purs_type"]
            if actual_strat == self.purs_types["pure_pursuit"]:
                tactic_obs[0] = 1.0
            elif actual_strat == self.purs_types["const_bear"]:
                tactic_obs[1] = 1.0
            elif actual_strat == self.purs_types["circling"]:
                tactic_obs[2] = 1.0
        is_overridden = 1.0 if getattr(self, 'override_active', False) else 0.0
        state_obs = np.array([is_attacking, cooldown_obs, is_overridden] + tactic_obs, dtype=np.float32)
        #MY STATE
        my_speed_mag = np.linalg.norm(self.curr_speed)
        my_acc_mag = np.linalg.norm(self.curr_acc)
        my_obs = np.concatenate([
            self.curr_speed / MAX_SPEED,      #velocity
            self.curr_acc / MAX_ACC,          #acceleration
            [my_speed_mag / MAX_SPEED],       #speed norm
            [my_acc_mag / MAX_ACC],           #acc norm
            [self.position[2] / MAX_COORD],   #height
            [self.my_rad / MAX_DRONE_RAD],    #radius
            [self.cruise_speed / MAX_SPEED],  #cruise velocity (maximum)
            [self.max_acc / MAX_ACC],         #acceleration maximum
            [self.curr_omega / MAX_ANG_VEL],  #angular speed
            [self.max_omega / MAX_ANG_VEL]    #angular maximum
        ]) 
        #PRIME STATE
        #radius
        prime_rad_obs = np.array([self.prime_rad / MAX_DRONE_RAD], dtype=np.float32)
        #rel velocity
        prime_rel_vel = self.prime_vel - self.curr_speed
        prime_pos = (self.prime_pos - self.position) / MAX_DIST
        #dist to me surface - surface
        raw_prime_dist = np.linalg.norm(self.prime_pos - self.position)
        surf_prime_dist = max(0.0, raw_prime_dist - self.prime_rad - self.my_rad)
        #normalizing
        prime_purs_dist = surf_prime_dist / MAX_DIST
        prime_obs = np.concatenate([
            prime_pos, 
            prime_rel_vel / (MAX_SPEED * 2.0),
            prime_rad_obs,
            [prime_purs_dist]
        ])
        #PURSER FILTER
        other_purs_pos = []
        other_purs_vel = []
        other_purs_rads = []
        other_purs_acc = []
        other_purs_ang_vel = []
        #in formation everyone is visible
        if self.state == States.FORM:
            other_purs_pos = list(self.all_purs_pos)
            other_purs_vel = list(self.all_purs_vel)
            other_purs_rads = list(self.all_purs_rads)
            other_purs_acc = list(self.all_purs_acc)
            other_purs_ang_vel = list(self.all_purs_ang_vel)  
        else:
            #in pursue only those also pursuing same target are relevant
            for i, tar in enumerate(self.all_purs_tars):
                if tar is not None and tar is self.target["target"]:
                    other_purs_pos.append(self.all_purs_pos[i])
                    other_purs_vel.append(self.all_purs_vel[i])
                    other_purs_rads.append(self.all_purs_rads[i])
                    other_purs_acc.append(self.all_purs_acc[i])
                    other_purs_ang_vel.append(self.all_purs_ang_vel[i])  
        new_purs_pos = np.array(other_purs_pos)
        new_purs_vel = np.array(other_purs_vel)
        new_purs_rads = np.array(other_purs_rads)
        new_purs_acc = np.array(other_purs_acc)
        new_purs_ang_vel = np.array(other_purs_ang_vel)
        #PURSUERS STATE
        pursuers_obs = np.full(42, FAR_AWAY, dtype=np.float32)
        #default
        for i in range(3):
            start = i * 14
            pursuers_obs[start+3 : start+6] = 0.0  #rel velocity
            pursuers_obs[start+6] = 0.0            #radius
            pursuers_obs[start+8 : start+11] = 0.0 #rel acc
            pursuers_obs[start+11] = 0.0           #angular speed
            pursuers_obs[start+12] = 0.0 #mate speed
            pursuers_obs[start+13] = 0.0 #mate acc mag
        density = 0
        if len(new_purs_pos) > 0:
            density = len(new_purs_pos)
            #normalized dist to me
            norm_rel_positions = (new_purs_pos - self.position) / self.vis_range
            norm_dists = np.linalg.norm(norm_rel_positions, axis=1)
            #three closest
            closest_indices = np.argsort(norm_dists)[:3]
            for i, idx in enumerate(closest_indices):
                start = i * 14
                #rel pos
                pursuers_obs[start : start+3] = norm_rel_positions[idx]    
                #rel velocity
                rel_vel = new_purs_vel[idx] - self.curr_speed
                pursuers_obs[start+3 : start+6] = rel_vel / (MAX_SPEED * 2.0)    
                #radius
                pursuers_obs[start+6] = new_purs_rads[idx] / MAX_DRONE_RAD    
                #dist to me surface - surface
                raw_mate_to_me_dist = np.linalg.norm(new_purs_pos[idx] - self.position)
                surf_mate_to_me_dist = max(0.0, raw_mate_to_me_dist - new_purs_rads[idx] - self.my_rad)
                pursuers_obs[start+7] = surf_mate_to_me_dist / self.vis_range
                #rel acceleration
                rel_acc = new_purs_acc[idx] - self.curr_acc
                pursuers_obs[start+8 : start+11] = rel_acc / (MAX_ACC * 2.0)
                #angular speed
                pursuers_obs[start+11] = new_purs_ang_vel[idx] / MAX_ANG_VEL
                #norm of rel speed
                mate_speed = np.linalg.norm(new_purs_vel[idx])
                pursuers_obs[start+12] = mate_speed / MAX_SPEED
                #norm of rel acc
                mate_acc_mag = np.linalg.norm(new_purs_acc[idx])
                pursuers_obs[start+13] = mate_acc_mag / MAX_ACC
        #DENSITY STATE
        density_obs = np.array([density / MAX_DENSITY], dtype=np.float32)
        #INVADER STATE
        self.current_tactical_invaders = []
        invaders_obs = np.full(36, FAR_AWAY, dtype=np.float32) 
        #default
        for i in range(2):
            start = i * 18
            invaders_obs[start+3 : start+6] = 0.0 #velocity
            invaders_obs[start+6] = 0.0           #cosinus
            invaders_obs[start+10] = 0.0          #num of pusuers
            invaders_obs[start+12:start+15] = 0.0 #rel acc
            invaders_obs[start+15] = 0.0          #norm speed
        if len(self.all_inv_pos) > 0:
            #two closest interest us
            # inv_dists = np.linalg.norm(self.all_inv_pos - self.position, axis=1)
            # sorted_inv_indices = np.argsort(inv_dists)
            tactical_scores = np.zeros(len(self.all_inv_pos), dtype=np.float32)
            #dist to prime
            raw_dists_to_prime = np.linalg.norm(self.all_inv_pos - self.prime_pos, axis=1)
            for idx in range(len(self.all_inv_pos)):
                #surface dist to Prime
                surf_dist_to_prime = max(0.0, raw_dists_to_prime[idx] - self.all_inv_rads[idx] - self.prime_rad)
                pursuers_count = self.all_inv_purs_num[idx]
                if surf_dist_to_prime < 20.0:
                    #critical zone, high priority
                    tactical_scores[idx] = surf_dist_to_prime + (pursuers_count * 1.0)
                else:
                    #safe dist, go after free invaders
                    tactical_scores[idx] = surf_dist_to_prime + (pursuers_count * 50.0)
            #sorting
            sorted_inv_indices = np.argsort(tactical_scores)
            #two tactically most important
            closest_inv_indices = sorted_inv_indices[:2]
            self.current_tactical_invaders = [self.targets[idx] for idx in closest_inv_indices]
            for i, idx in enumerate(closest_inv_indices):
                start = i * 18
                #rel position
                inv_rel_pos = (self.all_inv_pos[idx] - self.position) / MAX_DIST
                #rel velocity
                inv_rel_vel = self.all_inv_vel[idx] - self.curr_speed
                #cosinus shield
                me_to_prime = self.prime_pos - self.position
                me_to_inv = self.all_inv_pos[idx] - self.position
                dist_prime = np.linalg.norm(me_to_prime)
                dist_inv = np.linalg.norm(me_to_inv)
                if dist_prime > 0.001 and dist_inv > 0.001:
                    cos_angle = np.dot(me_to_prime, me_to_inv) / (dist_prime * dist_inv)
                else:
                    cos_angle = 0.0
                #dist to prime surface - surface
                raw_inv_prime_dist = np.linalg.norm(self.prime_pos - self.all_inv_pos[idx])
                surf_inv_prime_dist = max(0.0, raw_inv_prime_dist - self.all_inv_rads[idx] - self.prime_rad)
                #dist to me surface - surface
                raw_inv_purs_dist = np.linalg.norm(self.all_inv_pos[idx] - self.position)
                surf_inv_purs_dist = max(0.0, raw_inv_purs_dist - self.all_inv_rads[idx] - self.my_rad)
                #rel pos
                invaders_obs[start : start+3] = inv_rel_pos
                #rel velocity
                invaders_obs[start+3 : start+6] = inv_rel_vel / (MAX_SPEED * 2.0)
                #cosinus angle
                invaders_obs[start+6] = cos_angle
                #dist to prime
                invaders_obs[start+7] = surf_inv_prime_dist / MAX_COORD
                #dist to me
                invaders_obs[start+8] = surf_inv_purs_dist / MAX_COORD
                #radius
                invaders_obs[start+9] = self.all_inv_rads[idx] / MAX_DRONE_RAD
                #num of pursuers already chasing them
                invaders_obs[start+10] = self.all_inv_purs_num[idx] / OTHER_PURS  
                #if invader is targetable
                if surf_inv_prime_dist < 15.0:
                    max_attackers = 4
                else:
                    max_attackers = 2
                is_targetable = 1.0 if self.all_inv_purs_num[idx] < max_attackers else -1.0
                invaders_obs[start+11] = is_targetable
                #rel acc
                invaders_obs[start+12:start+15] = (self.all_inv_acc[idx] - self.curr_acc) / (MAX_ACC * 2.0)
                #norm of speed
                invaders_obs[start+15] = np.linalg.norm(self.all_inv_vel[idx] / MAX_SPEED)
                #norm of acc
                invaders_obs[start+16] = np.linalg.norm(self.all_inv_acc[idx] / MAX_ACC)
                #angular speed
                invaders_obs[start+17] = self.all_inv_ang_vel[idx] / MAX_ANG_VEL
        #OBSTACLES STATE
        obstacles_obs = np.full(20, FAR_AWAY, dtype=np.float32)
        #default
        obstacles_obs[3] = 0.0 
        obstacles_obs[8] = 0.0 
        obstacles_obs[13] = 0.0 
        obstacles_obs[18] = 0.0 
        obs_centers = self.obs_centers
        obs_radii = self.obs_radii
        if obs_centers is not None and len(obs_centers) > 0:
            #dist surface - surface
            center_dists = np.linalg.norm(obs_centers - self.position, axis=1)
            edge_dists = center_dists - obs_radii - self.my_rad
            #filtering those in sight
            visible_mask = edge_dists <= self.vis_range
            visible_indices = np.where(visible_mask)[0]
            if len(visible_indices) > 0:
                #sorting visible obstacles, four closest
                visible_edge_dists = edge_dists[visible_indices]
                sorted_local_indices = np.argsort(visible_edge_dists)[:4]
                closest_obs_indices = visible_indices[sorted_local_indices]
                #writing to obs space
                for i, idx in enumerate(closest_obs_indices):
                    start = i * 5
                    #rel pos
                    obs_pos = (obs_centers[idx] - self.position) / self.vis_range
                    obstacles_obs[start : start+3] = obs_pos
                    #radius
                    obstacles_obs[start+3] = obs_radii[idx] / MAX_RADIUS
                    #dist
                    obstacles_obs[start+4] = np.linalg.norm(obs_pos)
        #final vector
        final_obs = np.concatenate([
            state_obs,      # 6
            my_obs,         # 14
            prime_obs,      # 8
            density_obs,    # 1
            pursuers_obs,   # 42
            invaders_obs,   # 36
            obstacles_obs   # 20
        ]).astype(np.float32)
        return final_obs
    
    def get_observation_restrictive(self):
        #observation for herding NN, everything normalized
        MAX_COORD = 30.0       #world
        MAX_DIST = 40.0        #max possible distance
        MAX_SPEED = 8.5        #max speed
        MAX_DENSITY = 15.0     #max pursuer density
        OTHER_PURS = 5.0       #max pursuer density
        MAX_RADIUS = 5.0       #max obstacle radius
        MAX_DRONE_RAD = 0.7
        FAR_AWAY = 3.0         #far away, bigger number
        MAX_ACC = 8.0
        MAX_COOLDOWN = 5.0   
        #STATE STATE
        #flag is attacking
        is_attacking = 1.0 if self.state == States.PURSUE else 0.0
        #cooldown
        cooldown_obs = self.cooldown / MAX_COOLDOWN
        #tactic currently used
        tactic_obs = [0.0, 0.0, 0.0]
        if self.state == States.PURSUE and self.target is not None:
            actual_strat = self.target["purs_type"]
            if actual_strat == self.purs_types["pure_pursuit"]:
                tactic_obs[0] = 1.0
            elif actual_strat == self.purs_types["const_bear"]:
                tactic_obs[1] = 1.0
            elif actual_strat == self.purs_types["circling"]:
                tactic_obs[2] = 1.0
        is_overridden = 1.0 if getattr(self, 'override_active', False) else 0.0
        state_obs = np.array([is_attacking, cooldown_obs, is_overridden] + tactic_obs, dtype=np.float32)
        #MY STATE
        my_obs = np.concatenate([self.curr_speed / MAX_SPEED, self.curr_acc / MAX_ACC, [self.position[2] / MAX_COORD], 
                                 [self.my_rad / MAX_DRONE_RAD], [self.cruise_speed / MAX_SPEED], [self.max_speed / MAX_SPEED], [self.max_acc / MAX_ACC]]) 
        #PRIME STATE
        prime_rad_obs = np.array([self.prime_rad / MAX_DRONE_RAD], dtype=np.float32)
        prime_rel_vel = self.prime_vel - self.curr_speed
        prime_pos = (self.prime_pos - self.position) / MAX_DIST
        raw_prime_dist = np.linalg.norm(self.prime_pos - self.position)
        surf_prime_dist = max(0.0, raw_prime_dist - self.prime_rad - self.my_rad)
        prime_purs_dist = surf_prime_dist / MAX_DIST
        prime_obs = np.concatenate([prime_pos, prime_rel_vel / (MAX_SPEED * 2.0), prime_rad_obs, [prime_purs_dist]])
        #PURSER FILTER
        other_purs_pos = []
        other_purs_vel = []
        other_purs_rads = []
        #in formation everyone is visible
        if self.state == States.FORM:
            other_purs_pos = list(self.all_purs_pos)
            other_purs_vel = list(self.all_purs_vel)
            other_purs_rads = list(self.all_purs_rads)
        else:
            #in pursue only those also pursuing same target are relevant
            for i, tar in enumerate(self.all_purs_tars):
                if tar is not None and tar is self.target["target"]:
                    other_purs_pos.append(self.all_purs_pos[i])
                    other_purs_vel.append(self.all_purs_vel[i])
                    other_purs_rads.append(self.all_purs_rads[i])
        new_purs_pos = np.array(other_purs_pos)
        new_purs_vel = np.array(other_purs_vel)
        new_purs_rads = np.array(other_purs_rads)
        #PURSUERS STATE
        pursuers_obs = np.full(32, FAR_AWAY, dtype=np.float32)
        #default
        for i in range(4):
            start = i * 8
            pursuers_obs[start+3 : start+6] = 0.0  
            pursuers_obs[start+6] = 0.0  
        #density of pursuers          
        density = len(new_purs_pos)
        if density > 0:
            #sort them by distance
            norm_rel_positions = (new_purs_pos - self.position) / self.vis_range
            norm_dists = np.linalg.norm(norm_rel_positions, axis=1)
            closest_indices = np.argsort(norm_dists)[:4]
            for i, idx in enumerate(closest_indices):
                start = i * 8 
                #rel postion   
                pursuers_obs[start : start+3] = norm_rel_positions[idx]
                #rel velocity    
                rel_vel = new_purs_vel[idx] - self.curr_speed
                pursuers_obs[start+3 : start+6] = rel_vel / (MAX_SPEED * 2.0) 
                #radius   
                pursuers_obs[start+6] = new_purs_rads[idx] / MAX_DRONE_RAD    
                #dist to me surface - surface
                raw_mate_to_me_dist = np.linalg.norm(new_purs_pos[idx] - self.position)
                surf_mate_to_me_dist = max(0.0, raw_mate_to_me_dist - new_purs_rads[idx] - self.my_rad)
                pursuers_obs[start+7] = surf_mate_to_me_dist / self.vis_range
        #DENSITY STATE
        density_obs = np.array([density / MAX_DENSITY], dtype=np.float32)
        #INVADER STATE
        self.current_tactical_invaders = []
        invaders_obs = np.full(24, FAR_AWAY, dtype=np.float32) 
        #default
        for i in range(2):
            start = i * 12
            invaders_obs[start+3 : start+6] = 0.0 #velocity
            invaders_obs[start+6] = 0.0           #cosinus
            invaders_obs[start+10] = 0.0          #num of pusuers
        if len(self.all_inv_pos) > 0:
            #two closest interest us
            inv_dists = np.linalg.norm(self.all_inv_pos - self.position, axis=1)
            sorted_inv_indices = np.argsort(inv_dists)
            closest_inv_indices = sorted_inv_indices[:2]
            self.current_tactical_invaders = [self.targets[idx] for idx in closest_inv_indices]
            for i, idx in enumerate(closest_inv_indices):
                start = i * 12
                #rel position
                inv_rel_pos = (self.all_inv_pos[idx] - self.position) / MAX_COORD
                #rel velocity
                inv_rel_vel = self.all_inv_vel[idx] - self.curr_speed
                #cosinus shield
                me_to_prime = self.prime_pos - self.position
                me_to_inv = self.all_inv_pos[idx] - self.position
                dist_prime = np.linalg.norm(me_to_prime)
                dist_inv = np.linalg.norm(me_to_inv)
                if dist_prime > 0.001 and dist_inv > 0.001:
                    cos_angle = np.dot(me_to_prime, me_to_inv) / (dist_prime * dist_inv)
                else:
                    cos_angle = 0.0
                #dist to prime surface - surface
                raw_inv_prime_dist = np.linalg.norm(self.prime_pos - self.all_inv_pos[idx])
                surf_inv_prime_dist = max(0.0, raw_inv_prime_dist - self.all_inv_rads[idx] - self.prime_rad)
                #dist to me surface - surface
                raw_inv_purs_dist = np.linalg.norm(self.all_inv_pos[idx] - self.position)
                surf_inv_purs_dist = max(0.0, raw_inv_purs_dist - self.all_inv_rads[idx] - self.my_rad)
                #rel pos
                invaders_obs[start : start+3] = inv_rel_pos
                #rel velocity
                invaders_obs[start+3 : start+6] = inv_rel_vel / (MAX_SPEED * 2.0)
                #cosinus angle
                invaders_obs[start+6] = cos_angle
                #dist to prime
                invaders_obs[start+7] = surf_inv_prime_dist / MAX_COORD
                #dist to me
                invaders_obs[start+8] = surf_inv_purs_dist / MAX_COORD
                #radius
                invaders_obs[start+9] = self.all_inv_rads[idx] / MAX_DRONE_RAD
                #num of pursuers already chasing them
                invaders_obs[start+10] = self.all_inv_purs_num[idx] / OTHER_PURS  
                #if invader is targetable
                if surf_inv_prime_dist < 15.0:
                    max_attackers = 4
                else:
                    max_attackers = 2
                is_targetable = 1.0 if self.all_inv_purs_num[idx] < max_attackers else -1.0
                invaders_obs[start+11] = is_targetable
        #OBSTACLES STATE
        obstacles_obs = np.full(20, FAR_AWAY, dtype=np.float32)
        #default
        obstacles_obs[3] = 0.0 
        obstacles_obs[8] = 0.0 
        obstacles_obs[13] = 0.0 
        obstacles_obs[18] = 0.0 
        obs_centers = self.obs_centers
        obs_radii = self.obs_radii
        if obs_centers is not None and len(obs_centers) > 0:
            #dist surface - surface
            center_dists = np.linalg.norm(obs_centers - self.position, axis=1)
            edge_dists = center_dists - obs_radii - self.my_rad
            #filtering those in sight
            visible_mask = edge_dists <= self.vis_range
            visible_indices = np.where(visible_mask)[0]
            if len(visible_indices) > 0:
                #sorting visible obstacles, four closest
                visible_edge_dists = edge_dists[visible_indices]
                sorted_local_indices = np.argsort(visible_edge_dists)[:4]
                closest_obs_indices = visible_indices[sorted_local_indices]
                #writing to obs space
                for i, idx in enumerate(closest_obs_indices):
                    start = i * 5
                    #rel pos
                    obs_pos = (obs_centers[idx] - self.position) / self.vis_range
                    obstacles_obs[start : start+3] = obs_pos
                    #radius
                    obstacles_obs[start+3] = obs_radii[idx] / MAX_RADIUS
                    #dist
                    obstacles_obs[start+4] = np.linalg.norm(obs_pos)
        #final vector
        final_obs = np.concatenate([
            state_obs,      # 6
            my_obs,         # 11
            prime_obs,      # 8
            density_obs,    # 1
            pursuers_obs,   # 32
            invaders_obs,   # 24
            obstacles_obs   # 20
        ]).astype(np.float32)
        return final_obs
    
    def set_rl_action_restrictive(self, action_array, visible_invaders):
        self.is_rl_controlled = True
        self.tried_invalid_attack = False
        raw_attack_intent = action_array[11] > 0.0
        attack_intent = False
        tar = None
        is_already_my_target = False
        #Pursuer has target
        if self.target is not None and self.target["target"].crashed == False:
            tar = self.target["target"]
            is_already_my_target = True
            #retreat is not possible
            attack_intent = True 
        #Pursuer in formation
        else:
            #choosing new target
            if raw_attack_intent and len(visible_invaders) > 0:
                attack_intent = True
                target_idx = 0 if action_array[12] < 0.0 else min(1, len(visible_invaders) - 1)
                tar = visible_invaders[target_idx]
                self.override_active = False
            is_already_my_target = False
        #check max allowed pursuers if in attack
        if attack_intent and tar is not None:
            prime_inv_dist = np.linalg.norm(self.prime_pos - tar.position) - tar.my_rad - self.prime_rad
            max_allowed = 4 if prime_inv_dist < 15.0 else 2
            #how many other pursuers are pursuing
            others_attacking = tar.purs_num - 1 if is_already_my_target else tar.purs_num
            #too much pursuers, retreat
            if others_attacking >= max_allowed:
                attack_intent = False 
                if not is_already_my_target:
                    self.tried_invalid_attack = True
        #ATTACK
        if attack_intent and len(visible_invaders) > 0:
            self.state = States.PURSUE
            #choosing the target, if not having already
            if not is_already_my_target:
                target_idx = 0 if action_array[12] < 0.0 else min(1, len(visible_invaders) - 1)
                tar = visible_invaders[target_idx]
            #change of tactic
            strat_val = action_array[13]
            if not self.override_active:
                if strat_val < -0.33:
                    chosen_strategy = self.purs_types["circling"]
                elif strat_val < 0.33:
                    chosen_strategy = self.purs_types["const_bear"]
                else:
                    chosen_strategy = self.purs_types["pure_pursuit"]
            else:
                chosen_strategy = self.target["purs_type"]
            #saving changes
            if self.target is None or self.target["target"] is not tar:
                if self.target is not None:
                    self.target["target"].purs_num -= 1
                tar.purs_num += 1 
            #save the target
            self.target = {"target": tar, "tar_pos": tar.position, 
                            "tar_vel": tar.curr_speed, "tar_rad": tar.my_rad, "tar_ang": tar.curr_omega, "tar_acc": tar.curr_acc, "purs_type": chosen_strategy}
        #FORMATION
        else:
            self.state = States.FORM
            if self.target is not None:
                self.target["target"].purs_num -= 1
            self.target = None
        #micro decisions
        if self.state == States.PURSUE:
            self.rep_in_purs = np.interp(action_array[0], [-1, 1], [0.1, 2.0])
            self.purs = np.interp(action_array[1], [-1, 1], [0.5, 3.0])
            self.coll_obs = np.interp(action_array[2], [-1, 1], [0.5, 3.0]) 
            self.rep_obs = np.interp(action_array[3], [-1, 1], [0.5, 3.0])  
            self.prime_rep_in_purs = np.interp(action_array[4], [-1, 1], [2.5, 5.0])
            self.prime_coll_r = np.interp(action_array[5], [-1, 1], [2.0, 6.0])
            self.collision_r = np.interp(action_array[7], [-1, 1], [0.5, 3.0])
        elif self.state == States.FORM:
            self.override_active = False
            self.rep_in_form = np.interp(action_array[0], [-1, 1], [4.5, 10.0])
            self.form = np.interp(action_array[1], [-1, 1], [0.5, 5.0])
            self.coll_obs = np.interp(action_array[2], [-1, 1], [2.0, 10.0]) 
            self.rep_obs = np.interp(action_array[3], [-1, 1], [0.5, 5.0])   
            self.formation_r = np.interp(action_array[4], [-1, 1], [1.0, 3.0])
            self.formation_r_min = np.interp(action_array[5], [-1, 1], [0.5, self.formation_r])
            self.rep_invs_r = np.interp(action_array[6], [-1, 1], [3.0, 15.0])
            self.collision_r = np.interp(action_array[7], [-1, 1], [2.0, 3.0])
        #joint
        #self.collision_r = np.interp(action_array[7], [-1, 1], [2.5, 3.0])
        self.coll_gr = np.interp(action_array[8], [-1, 1], [0.5, 3.0])
        self.rep_gr = np.interp(action_array[9], [-1, 1], [0.5, 4.0])
        if self.target is not None and self.target["purs_type"] != self.purs_types["circling"]:
            self.cruise_speed = self.max_speed*0.75#np.interp(action_array[10], [-1, 1], [max(5.0, self.max_speed*0.75 - 1.5), self.max_speed*0.75])
        else:
            self.cruise_speed = np.interp(action_array[10], [-1, 1], [0.1, self.max_speed*0.75])
        self.KP = np.interp(action_array[14], [-1, 1], [2.0, 7.0])
        self.KD = np.interp(action_array[15], [-1, 1], [0.0, 0.5])

    def set_rl_action(self, action_array, visible_invaders):
        self.is_rl_controlled = True
        self.tried_invalid_attack = False
        #reset of cooldown
        if self.target is not None:
            dist_to_prime = np.linalg.norm(self.prime_pos - self.target["tar_pos"]) - self.target["tar_rad"] - self.prime_rad
            if dist_to_prime < 15.0:
                self.cooldown = 0    
        #controlling cooldown
        if self.cooldown > 0:
            self.cooldown -= 1
            pass 
        else:
            raw_attack_intent = action_array[11] > 0.0
            #emergancy override
            if getattr(self, 'override_active', False) and self.target is not None and self.target["target"].crashed == False:
                attack_intent = True
                tar = self.target["target"]
                is_already_my_target = True 
            #normal state
            else:
                if raw_attack_intent and len(visible_invaders) > 0:
                    attack_intent = True
                    target_idx = 0 if action_array[12] < 0.0 else min(1, len(visible_invaders) - 1)
                    tar = visible_invaders[target_idx]
                    #same target
                    if self.target is not None and self.target["target"] is tar:
                        pass
                    #new target
                    else:
                        self.override_active = False 
                else:
                    #retreat from the attack
                    attack_intent = False
                    tar = None
                    self.override_active = False
            #ATTACK
            if attack_intent and len(visible_invaders) > 0:
                self.state = States.PURSUE
                #choosing tactic
                strat_val = action_array[13]
                if not getattr(self, 'override_active', False):
                    if strat_val < -0.33:
                        chosen_strategy = self.purs_types["circling"]
                    elif strat_val < 0.33:
                        chosen_strategy = self.purs_types["const_bear"]
                    else:
                        chosen_strategy = self.purs_types["pure_pursuit"]
                else:
                    chosen_strategy = self.target["purs_type"] if self.target else self.purs_types["const_bear"]    
                #change parameters and cooldown
                if self.target is None or self.target["target"] is not tar:
                    if self.target is not None:
                        self.target["target"].purs_num -= 1
                    tar.purs_num += 1
                    self.cooldown = 5
                #new tactic, also cooldown
                elif self.target is not None and self.target["purs_type"] != chosen_strategy:
                    self.cooldown = 5
                #saving the state
                self.target = {"target": tar, "tar_pos": tar.position, 
                                "tar_vel": tar.curr_speed, "tar_rad": tar.my_rad, "tar_ang": tar.curr_omega, "tar_acc": tar.curr_acc, "purs_type": chosen_strategy}
            #FORMATION
            else:
                #back to formation, cooldown
                if self.state == States.PURSUE:
                    self.cooldown = 5
                self.state = States.FORM
                if self.target is not None:
                    self.target["target"].purs_num -= 1
                self.target = None
        #micro decision
        if self.state == States.PURSUE:
            self.rep_in_purs = np.interp(action_array[0], [-1, 1], [0.1, 2.0])
            self.purs = np.interp(action_array[1], [-1, 1], [0.5, 3.0])
            self.coll_obs = np.interp(action_array[2], [-1, 1], [0.5, 3.0]) 
            self.rep_obs = np.interp(action_array[3], [-1, 1], [0.5, 3.0])  
            self.prime_rep_in_purs = np.interp(action_array[4], [-1, 1], [2.5, 5.0])
            self.prime_coll_r = np.interp(action_array[5], [-1, 1], [2.0, 6.0])
            self.collision_r = np.interp(action_array[7], [-1, 1], [0.5, 3.0])
        elif self.state == States.FORM:
            self.override_active = False
            self.rep_in_form = np.interp(action_array[0], [-1, 1], [4.5, 10.0])
            self.form = np.interp(action_array[1], [-1, 1], [0.5, 5.0])
            self.coll_obs = np.interp(action_array[2], [-1, 1], [2.0, 10.0]) 
            self.rep_obs = np.interp(action_array[3], [-1, 1], [0.5, 5.0])   
            self.formation_r = np.interp(action_array[4], [-1, 1], [1.0, 3.0])
            self.formation_r_min = np.interp(action_array[5], [-1, 1], [0.5, self.formation_r])
            self.rep_invs_r = np.interp(action_array[6], [-1, 1], [3.0, 15.0])
            self.collision_r = np.interp(action_array[7], [-1, 1], [2.0, 3.0])
        #joint params
        self.coll_gr = np.interp(action_array[8], [-1, 1], [0.5, 3.0])
        self.rep_gr = np.interp(action_array[9], [-1, 1], [0.5, 4.0])
        if self.target is not None and self.target["purs_type"] != self.purs_types["circling"]:
            self.cruise_speed = self.max_speed*0.75 #np.interp(action_array[10], [-1, 1], [max(5.0, self.max_speed*0.75 - 1.5), self.max_speed*0.75])
        else:
            self.cruise_speed = np.interp(action_array[10], [-1, 1], [0.1, self.max_speed*0.75])
        self.KP = np.interp(action_array[14], [-1, 1], [2.0, 7.0])
        self.KD = np.interp(action_array[15], [-1, 1], [0.0, 0.5])
    
    def copy_data(self, prime_vel, prime_pos):
        #copies all data, that are going to be modified
        self.prime_vel = prime_vel.copy()
        self.prime_pos = prime_pos.copy()
        
    def update_target(self):
        #updates parameters of target
        self.target["tar_pos"] = self.target["target"].position.copy()
        self.target["tar_rad"] = self.target["target"].my_rad
        self.target["tar_vel"] = self.target["target"].curr_speed.copy()
        self.target["tar_acc"] = self.target["target"].curr_acc.copy()
        self.target["tar_ang"] = self.target["target"].curr_omega
    
    def gauss_noise(self):
        #apply gauss
        # if len(self.all_inv_pos) != 0:
        #     self.apply_gaussian_noise(self.all_inv_pos, self.all_inv_rads, self.all_inv_vel)    
        # if self.obs_centers is not None and len(self.obs_centers) > 0:
        #     self.apply_gaussian_noise(self.obs_centers, rads=self.obs_radii)    
        #if len(self.every_purs_pos) != 0:
        #    self.apply_gaussian_noise(self.every_purs_pos, velocities=self.every_purs_vel)
        #target
        if self.target is not None:
            dist = np.linalg.norm(self.position - self.target["tar_pos"])
            self.target["tar_pos"] += np.random.normal(0.0, self.base_pos + self.k_pos * dist, self.pos_length)
            self.target["tar_vel"] += np.random.normal(0.0, self.base_vel + self.k_vel * dist, self.pos_length)
            new_rad = self.target["tar_rad"] + np.random.normal(0.0, self.base_rad + self.k_rad * dist)
            self.target["tar_rad"] = max(0.01, float(new_rad))  
            self.target["tar_acc"] += np.random.normal(0.0, self.base_acc + self.k_acc * dist, self.pos_length)
            self.target["tar_ang"] += np.random.normal(0.0, self.base_ang + self.k_ang * dist) 
        #Prime
        dist = np.linalg.norm(self.position - self.prime_pos)
        self.prime_pos += np.random.normal(0.0, self.base_pos + self.k_pos * dist, self.pos_length)
        self.prime_vel += np.random.normal(0.0, self.base_vel + self.k_vel * dist, self.pos_length)
        new_rad = self.prime_rad + np.random.normal(0.0, self.base_rad + self.k_rad * dist)
        self.prime_rad = max(0.01, float(new_rad))
    
    def apply_gaussian_noise(self, positions, rads=None, velocities=None, accs=None, ang_vels=None):
        #all dists at once
        diffs = positions - self.position
        dists_center = np.linalg.norm(diffs, axis=1)
        #sigmas pos
        sigmas_pos = self.base_pos + self.k_pos * dists_center
        #pos noise
        positions += np.random.normal(loc=0.0, scale=sigmas_pos[:, np.newaxis], size=positions.shape)
        #rads
        if rads is not None:
            sigmas_rad = self.base_rad + self.k_rad * dists_center
            noise_rad = np.random.normal(loc=0.0, scale=sigmas_rad)
            rads[:] = np.clip(rads + noise_rad, a_min=0.01, a_max=None)
        #velocity
        if velocities is not None:
            sigmas_vel = self.base_vel + self.k_vel * dists_center
            velocities += np.random.normal(loc=0.0, scale=sigmas_vel[:, np.newaxis], size=velocities.shape)
        #acceleration
        if accs is not None:
            sigmas_acc = self.base_acc + self.k_acc * dists_center
            accs += np.random.normal(loc=0.0, scale=sigmas_acc[:, np.newaxis], size=accs.shape)
        #angular velocity
        if ang_vels is not None:
            sigmas_ang = self.base_ang + self.k_ang * dists_center
            ang_vels += np.random.normal(loc=0.0, scale=sigmas_ang, size=ang_vels.shape)
        return positions, rads, velocities, accs, ang_vels

    def filter_visible_objects(self, positions, velocities, rads, max_range, apply_noise=False, obstacles=False, accs=None, ang_vels=None):
        #empty mask
        empty_mask = np.array([], dtype=bool)
        #early return
        if len(positions) == 0:
            #None for obstacles
            if obstacles:
                return None, None, None, None, None, empty_mask
            #arrays for invaders
            else:
                ret_vel = np.empty((0, self.pos_length)) if velocities is not None else None
                ret_acc = np.empty((0, self.pos_length)) if accs is not None else None
                ret_ang = np.empty((0,)) if ang_vels is not None else None
                return np.empty((0, self.pos_length)), ret_vel, np.empty((0,)), ret_acc, ret_ang, empty_mask
        #dist and noise
        diffs = positions - self.position
        dists_center = np.linalg.norm(diffs, axis=1)
        if apply_noise:
            positions, rads, velocities, accs, ang_vels = self.apply_gaussian_noise(positions, rads, velocities, accs, ang_vels)
            diffs = positions - self.position
            dists_center = np.linalg.norm(diffs, axis=1)
        dists_surface = dists_center - self.my_rad - rads
        #mask
        mask = dists_surface <= max_range
        #apply mask
        visible_pos = positions[mask]
        visible_rad = rads[mask]
        visible_vel = velocities[mask] if velocities is not None else None
        visible_acc = accs[mask] if accs is not None else None
        visible_ang = ang_vels[mask] if ang_vels is not None else None
        #return
        if obstacles and len(visible_pos) == 0:
            return None, None, None, None, None, mask
        return visible_pos, visible_vel, visible_rad, visible_acc, visible_ang, mask
    
    def get_visible_neighbors_data(self, all_pos, all_vel, all_rad, all_tars, all_acc, all_ang, gaussian=False):
        #dists to all
        diffs = all_pos - self.position
        dists_center = np.linalg.norm(diffs, axis=1)
        if gaussian:
            all_pos, all_rad, all_vel, all_acc, all_ang = self.apply_gaussian_noise(all_pos, all_rad, all_vel, all_acc, all_ang)    
        #trick, my distance is infinity
        dists_center[self.my_index] = np.inf
        #mask visibility
        dists_surface = dists_center - self.my_rad - all_rad
        mask = dists_surface < self.vis_range
        #only those visible
        if not np.any(mask):
             return np.empty((0, self.pos_length)), np.empty((0, self.pos_length)), np.empty((0,)), np.empty((0, self.pos_length)), np.empty((0,)), [] 
        visible_pos = all_pos[mask]
        visible_rad = all_rad[mask]
        visible_vel = all_vel[mask]
        visible_acc = all_acc[mask]
        visible_ang = all_ang[mask]
        visible_tar = [tar for tar, m in zip(all_tars, mask) if m]
        return visible_pos, visible_vel, visible_rad, visible_acc, visible_ang, visible_tar
    
    def strategy_capture_cone(self, targets: list[Invader], sim_time: float, cooldown: float = 10.0):
        if not targets:
            return False
        if not hasattr(self, "ignored_targs"):
            self.ignored_targs = {}
        #invader pos
        t_pos = self.all_inv_pos
        #those invaders, that are close enough to unit
        near_mask = np.linalg.norm(t_pos - self.prime_pos, axis=1) < self.capture_r
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
                dist = np.linalg.norm(inv.position - self.prime_pos)
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
        unit_vec = self.prime_pos - self.position
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
            self.target = {"target": targets[best_idx], "tar_pos": self.all_inv_pos[best_idx], "tar_vel": self.all_inv_vel[best_idx],"tar_rad": self.all_inv_rads[best_idx], 
                           "tar_ang": self.all_inv_ang_vel[best_idx], "tar_acc": self.all_inv_acc[best_idx], "purs_type": self.purs_types['circling']}
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
            self.target = {"target" :targets[idx], "tar_pos": self.all_inv_pos[idx], "tar_vel": self.all_inv_vel[idx], "tar_rad": self.all_inv_rads[idx], 
                           "tar_ang": self.all_inv_ang_vel[idx], "tar_acc": self.all_inv_acc[idx], "purs_type": self.purs_types['circling']}
            self.state = States.PURSUE
            self.ignored_targs.clear()
            return True
        return False 
    
    def repulsive_force_ground(self, coll):
        total_force = np.zeros_like(self.position)
        if len(total_force) == 2:
            return total_force
        if self.position[2] - self.my_rad < coll:
            magnitude = (1.0 / self.position[2]) - (1.0 / coll)
            rep_dir = np.array([0, 0, 1])
            total_force = rep_dir * magnitude * self.cruise_speed
        #total force
        return total_force
    
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
        return np.sum(push_dirs * magnitudes[:, np.newaxis], axis=0) * self.cruise_speed

    def repulsive_force_prime(self, coll: float):
        total_force = np.zeros_like(self.position)
        #distance center to center
        diff = self.position - self.prime_pos
        dist_center = np.linalg.norm(diff)
        if dist_center < 1e-6:
            dist_center = 1e-6
        #distance of surfaces
        dist_surface = dist_center - self.my_rad - self.prime_rad
        #computing the force, if close enough
        if dist_surface < coll:
            push_dir = diff/dist_center
            magnitude = (1.0 / dist_surface) - (1.0 / coll)
            total_force = push_dir * magnitude
        return total_force * self.cruise_speed
    
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
    
    def repulsive_inv_force(self, targets):
        rep_dir = np.zeros_like(self.position)
        close_targs = []
        dists = []
        #targets that are close
        if targets:
            t_pos = self.all_inv_pos
            t_rad = self.all_inv_rads
            #dists center to center
            vecs_to_t = t_pos - self.prime_pos
            dists_c = np.linalg.norm(vecs_to_t, axis=1)
            #zero div protection
            dists_c[dists_c < 1e-6] = 1e-6
            #dists surface to surface
            dists_s = dists_c - t_rad - self.prime_rad
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
        #making weighted center of mass and then repulsive force against
        if close_targs:
            close_targs = np.array(close_targs)
            dists = np.array(dists)
            #weights, the closer, the bigger
            exp_dists = np.exp(-dists)
            weights = exp_dists / np.sum(exp_dists)
            w_center_of_mass = np.sum(close_targs * weights[:, np.newaxis], axis=0)
            #calculation rel pos of pursuer, finding out if pursuer is in between prime and center of mass
            vec_prime_to_pursuer = self.position - self.prime_pos
            vec_prime_to_com = w_center_of_mass - self.prime_pos
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
            diff = self.prime_pos - w_center_of_mass
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
        return rep_dir * self.cruise_speed
        
    def sigmoid(self, x):
        #sigmoid function
        x_safe = np.clip(x, -250.0, 250.0)
        return 1 / (1 + np.exp(-x_safe))
    
    def form_vortex_field_circle(self, mock_position=None):
        if mock_position is not None:
            my_pos = mock_position
        else:
            my_pos = self.position
        unit_pos = self.prime_pos
        unit_vel = self.prime_vel
        #the center of the vortex directly at prime
        rel_unit_pos = my_pos - unit_pos
        #radius is combination of two according to prime fly direction
        sigm = self.sigmoid(np.dot(unit_vel, rel_unit_pos))
        form_r = sigm * self.formation_r + (1 - sigm) * self.formation_r_min
        dist = np.linalg.norm(rel_unit_pos) - self.prime_rad - self.my_rad
        if dist < 1e-6:
            return np.zeros_like(my_pos)
        rho = 1 - (dist / form_r)**2
        #inside of circle
        if rho > 0:
            alpha = 2.5 * self.cruise_speed
        #outside of circle
        else:
            alpha = 1.0
        norm_vec = rel_unit_pos/np.linalg.norm(rel_unit_pos)
        #circle around center
        tangent_vec = np.array([-rel_unit_pos[1], rel_unit_pos[0]])
        tan_len = np.linalg.norm(tangent_vec)
        if tan_len > self.circle_tan_max:
            tangent_vec = (tangent_vec / tan_len) * self.circle_tan_max
        #feedbackward
        fdbck = alpha * rho * norm_vec
        #feedforward
        circle_dir = self.circle_dir
        fdwrd = circle_dir * tangent_vec
        form_vel = fdbck + fdwrd
        #normalizing
        if rho < 0:
            form_norm = np.linalg.norm(form_vel)
            if form_norm > self.max_form_speed:
                form_vel = (form_vel/form_norm)*self.max_form_speed
        return form_vel
    
    def calculate_axis_consensus(self, nei_pos, nei_vel, center_pos):
        if len(nei_pos) == 0:
            return None
        #sum of all axes
        #positions and speeds from everyone
        n_pos = nei_pos
        n_speed = nei_vel
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

    def get_axis_at_time(self, t):
        #school of fish
        time_scaled = t * 0.3
        axis = np.array([np.sin(time_scaled), np.cos(time_scaled * 0.7), np.sin(time_scaled * 1.2)])
        norm = np.linalg.norm(axis)
        if norm > 1e-6:
            return axis / norm
        return np.array([0.0, 0.0, 1.0])
    
    def get_final_axis(self, current_axis, observed_group_axis):
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
        return final_axis

    def form_vortex_field_sphere(self, mock_position=None):
        if mock_position is not None:
            my_pos = mock_position
        else:
            my_pos = self.position
        unit_pos = self.prime_pos
        unit_vel = self.prime_vel
        rel_unit_pos = my_pos - unit_pos
        #the radius is according to the position of pursuer w.r.t. prime's velocity vector
        sigm = self.sigmoid(np.dot(unit_vel, rel_unit_pos))
        form_r = sigm * self.formation_r + (1 - sigm) * self.formation_r_min
        #distance from prime
        dist = np.linalg.norm(rel_unit_pos) - self.prime_rad - self.my_rad
        if dist < 1e-6:
            return np.zeros_like(my_pos)
        rho = 1 - (dist / form_r)**2
        #alpha, stronger on the inside
        if rho > 0:
            alpha = 2.5 * self.cruise_speed
        else:
            alpha = 1.0
        #normalized vec from prime to pursuer
        normal_vec = rel_unit_pos / np.linalg.norm(rel_unit_pos)
        #observed average axis of neighbors
        observed_group_axis = self.calculate_axis_consensus(self.all_purs_pos, self.all_purs_vel, unit_pos)
        #my axis
        current_axis = self.get_axis_at_time(self.num_iter)
        #synchronization of clocks
        if observed_group_axis is not None:
            final_axis = self.get_final_axis(current_axis, observed_group_axis)
        else:
            #being alone
            final_axis = current_axis
        #resulting tangent
        tangent_vec = np.cross(final_axis, rel_unit_pos)
        tan_len = np.linalg.norm(tangent_vec)
        if tan_len > 1e-6:
            tangent_vec = tangent_vec / tan_len
        #composing the forces into resulting form vector
        fdbck = alpha * rho * normal_vec
        fdwrd = self.circle_dir * tangent_vec * self.max_speed
        form_vel = fdwrd + fdbck
        if rho < 0:
            form_norm = np.linalg.norm(form_vel)
            if form_norm > self.max_form_speed:
                form_vel = (form_vel/form_norm)*self.max_form_speed
        return form_vel
    
    def pursue_rl_target(self, target):
        tar_speed = np.linalg.norm(target["target"].curr_speed) #np.linalg.norm(target["tar_vel"])
        my_speed = self.max_speed * 0.75
        #if target is faster then pursuer, just pure pursue him
        if tar_speed >= my_speed or target["purs_type"] == self.purs_types['pure_pursuit']:
            if tar_speed >= my_speed:
                self.override_active = True
                #print("override active")
            target["purs_type"] = self.purs_types['pure_pursuit']
            return self.pursuit_pure_pursuit(target)
        #still too fast for encirclement, CB him
        elif tar_speed >= my_speed*0.85 or target["purs_type"] == self.purs_types['const_bear']:
            if tar_speed >= my_speed*0.85:
                self.override_active = True
                #print("override active")
            target["purs_type"] = self.purs_types['const_bear']
            return self.pursuit_constant_bearing(target)
        else:
            self.override_active = False
            return self.pursue_herding()
    
    def pursue_target(self, target):
        tar_speed = np.linalg.norm(target["target"].curr_speed) #np.linalg.norm(target["tar_vel"])
        my_speed = self.cruise_speed
        prime_inv_dist = np.linalg.norm(self.prime_pos - target["tar_pos"])
        #if target is faster then pursuer, just pure pursue him
        if tar_speed >= my_speed or target["purs_type"] == self.purs_types['pure_pursuit'] or (prime_inv_dist <= 15.0 and tar_speed >= my_speed):
            if prime_inv_dist > 15.0:
                target["purs_type"] = self.purs_types['pure_pursuit']
            else:
                target["purs_type"] = self.purs_types['pure_pursuit1']
            return self.pursuit_pure_pursuit(target)
        #still too fast for encirclement, CB him
        elif tar_speed >= my_speed/1.2 or target["purs_type"] == self.purs_types['const_bear'] or prime_inv_dist <= 15.0:
            if prime_inv_dist > 15.0:
                target["purs_type"] = self.purs_types['const_bear']
            else:
                target["purs_type"] = self.purs_types['const_bear1']
            return self.pursuit_constant_bearing(target)
        #if more then one is chasing him and he is further from unit, circle him
        if np.linalg.norm(self.prime_pos - target["tar_pos"]) >= self.safe_circle_r:
            # for p in purs:
            #     if p is not self and p.target != None and p.target[0] is target[0]:
            if target["target"].purs_num >= 2:
                if self.pos_length == 2:
                    return self.pursuit_circling(target)
                else:
                    return self.pursuit_sphering(target)
        #no one else is chasing him, catch him
        target["purs_type"] = self.purs_types['const_bear1']
        return self.pursuit_constant_bearing(target)
    
    def pursuit_circling(self, target, mock_position=None):
        if mock_position is not None:
            my_pos = mock_position
        else:
            my_pos = self.position
        target["purs_type"] = self.purs_types['circling']
        #strong repulsive force is needed
        if not self.is_rl_controlled:
            self.prime_coll_r = 10.5
            self.rep_in_purs = 4.8
            self.prime_rep_in_purs = 4.4
        #the center of the vortex field is not shifted
        rel_pos = my_pos - (target["tar_pos"]) #+ target[0].curr_speed * self.dt * self.pred_time)
        dist = np.linalg.norm(rel_pos) - target["tar_rad"] - self.my_rad
        if dist < 1e-6:
            return np.zeros_like(my_pos)
        rho = 1 - (dist / self.t_circle)**2
        #inside of circle
        if rho > 0:
            alpha = 2.5 * self.cruise_speed
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
        vel_dot = np.dot(target["tar_vel"], self.curr_speed)
        if self.t_circle * 9.0 >= dist >= self.t_circle * 3.0 and vel_norm >= 3.0 and vel_dot < 0:
            purs_vel = purs_vel/vel_norm * 0.9
        return purs_vel
    
    def pursuit_sphering(self, target):
        #other pursuers pursuing same target in visibility region
        other_purs_pos = []
        other_purs_vel = []
        for i, tar in enumerate(self.all_purs_tars):
            if tar is not None and tar is target["target"]:
                other_purs_pos.append(self.all_purs_pos[i])
                other_purs_vel.append(self.all_purs_vel[i])
        target["purs_type"] = self.purs_types['circling']
        #strong repulsive force is needed
        if not self.is_rl_controlled:
            self.prime_coll_r = 10.5
            self.rep_in_purs = 4.8
            self.prime_rep_in_purs = 4.4
        my_pos = self.position
        rel_pos = my_pos - (target["tar_pos"])
        #distance from target
        dist = np.linalg.norm(rel_pos) - target["tar_rad"] - self.my_rad
        if dist < 1e-6:
            return np.zeros_like(my_pos)
        rho = 1 - (dist / self.t_circle)**2
        #alpha, stronger on the inside
        if rho > 0:
            alpha = 10.5 * self.cruise_speed
        else:
            alpha = 1.0
        #normalized vec from target to pursuer
        normal_vec = rel_pos / np.linalg.norm(rel_pos)
        #observed average axis of neighbors
        observed_group_axis = self.calculate_axis_consensus(np.array(other_purs_pos), np.array(other_purs_vel), self.prime_pos)
        #my axis
        current_axis = self.get_axis_at_time(self.num_iter)
        #synchronization of clocks
        if observed_group_axis is not None:
            final_axis = self.get_final_axis(current_axis, observed_group_axis)
        else:
            #being alone
            final_axis = current_axis
        #resulting tangent
        tangent_vec = np.cross(final_axis, rel_pos)
        tan_norm = np.linalg.norm(tangent_vec)
        if tan_norm > 1e-6:
            tangent_vec = tangent_vec / tan_norm
        #composing the forces into resulting form vector
        fdbck = alpha * rho * normal_vec
        fdwrd = self.circle_dir * tangent_vec
        purs_vel = fdwrd + fdbck
        purs_norm = np.linalg.norm(purs_vel)
        if purs_norm > self.cruise_speed:
            purs_vel = (purs_vel/purs_norm)*self.cruise_speed
        return purs_vel
        
    def pursuit_constant_bearing(self, target):
        #target is quite fast, circling is not possible
        if not self.is_rl_controlled:
            self.prime_coll_r = 1.5
            self.rep_in_purs = 1.3
            self.prime_rep_in_purs = 2.5
        v_tar = target["tar_vel"]
        #line of sight
        r = target["tar_pos"] - self.position
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
        CB_dir = CB_dir * self.cruise_speed
        return CB_dir
    
    def pursuit_pure_pursuit(self, target):
        #target is very fast, pure pursuit
        if not self.is_rl_controlled:
            self.prime_coll_r = 1.5
            self.rep_in_purs = 1.3
            self.prime_rep_in_purs = 2.5
        PP_dir = target["tar_pos"] - self.position
        #norming it to the max speed
        if np.linalg.norm(PP_dir) < 1e-12:
            return np.zeros_like(PP_dir)
        PP_dir = PP_dir / np.linalg.norm(PP_dir)
        PP_dir = PP_dir * self.cruise_speed
        return PP_dir
    
# =============== CURRENTLY UNUSED CODE ======================

    # def observation_herding(self):
    #     #observation for herding NN, everything normalized
    #     MAX_COORD = 30.0       #world
    #     MAX_DIST = 40.0        #max possible distance
    #     MAX_SPEED = 8.5        #max speed
    #     MAX_DENSITY = 6.0     #max pursuer density
    #     MAX_RADIUS = 5.0      #max obstacle radius
    #     MAX_DRONE_RAD = 0.7
    #     FAR_AWAY = 3.0  #far away, bigger number
    #     MAX_ACC = 8.0
    #     #my state
    #     my_obs = np.concatenate([self.curr_speed / MAX_SPEED, self.curr_acc / MAX_ACC, [self.position[2] / MAX_COORD], 
    #                              [self.my_rad / MAX_DRONE_RAD], [self.cruise_speed / MAX_SPEED], [self.max_acc / MAX_ACC]]) 
    #     # prime state
    #     prime_rad_obs = np.array([self.prime_rad / MAX_DRONE_RAD], dtype=np.float32)
    #     prime_rel_vel = self.prime_vel - self.curr_speed
    #     prime_pos = (self.prime_pos - self.position) / MAX_DIST
    #     # 1. Výpočet reálné vzdálenosti mezi středy MĚ a PRIME
    #     raw_prime_dist = np.linalg.norm(self.prime_pos - self.position)
    #     # 2. Odečtení poloměrů (Povrchová vzdálenost)
    #     surf_prime_dist = max(0.0, raw_prime_dist - self.prime_rad - self.my_rad)
    #     # 3. Normalizace pro Observation Space
    #     prime_purs_dist = surf_prime_dist / MAX_DIST
    #     prime_obs = np.concatenate([
    #         prime_pos, 
    #         prime_rel_vel / (MAX_SPEED * 2.0),
    #         prime_rad_obs,
    #         [prime_purs_dist]
    #     ])
    #     #dividing pursuers to those chasing invader with me and the others
    #     other_purs_pos = []
    #     other_purs_vel = []
    #     other_purs_rads = []
    #     # new_obs_pos = []
    #     # new_obs_vel = []
    #     # new_obs_rad = []
    #     for i, tar in enumerate(self.all_purs_tars):
    #         if tar is not None and tar is self.target["target"]:
    #             other_purs_pos.append(self.all_purs_pos[i])
    #             other_purs_vel.append(self.all_purs_vel[i])
    #             other_purs_rads.append(self.all_purs_rads[i])
    #         # else:
    #         #     new_obs_pos.append(self.all_purs_pos[i])
    #         #     new_obs_rad.append(self.all_purs_rads[i])
    #         #     new_obs_vel.append(self.all_purs_vel[i])
    #     new_purs_pos = np.array(other_purs_pos)
    #     new_purs_vel = np.array(other_purs_vel)
    #     new_purs_rads = np.array(other_purs_rads)
    #     # Získáme pozici Invadera pro výpočet
    #     invader_pos = self.target["tar_pos"]
    #     # Zvětšíme pole: 4 kolegové * 9 informací = 36 (místo 32)
    #     pursuers_obs = np.full(36, FAR_AWAY, dtype=np.float32)
    #     # Defaultní hodnoty (nuly pro rychlost a poloměr)
    #     for i in range(4):
    #         start = i * 9  # <-- Změna multiplikátoru na 9
    #         pursuers_obs[start+3 : start+6] = 0.0  # default velocity
    #         pursuers_obs[start+6] = 0.0            # default radius
    #     density = 0
    #     if len(new_purs_pos) > 0:
    #         density = len(new_purs_pos)
    #         # Normalizované pozice vůči mně
    #         norm_rel_positions = (new_purs_pos - self.position) / self.vis_range
    #         norm_dists = np.linalg.norm(norm_rel_positions, axis=1)
    #         # Seřazení podle vzdálenosti (kdo je mi nejblíž)
    #         closest_indices = np.argsort(norm_dists)[:4]
    #         for i, idx in enumerate(closest_indices):
    #             start = i * 9  # <-- Změna multiplikátoru na 9
    #             # 0, 1, 2: Relativní pozice
    #             pursuers_obs[start : start+3] = norm_rel_positions[idx]    
    #             # 3, 4, 5: Relativní rychlost
    #             rel_vel = new_purs_vel[idx] - self.curr_speed
    #             pursuers_obs[start+3 : start+6] = rel_vel / (MAX_SPEED * 2.0)    
    #             # 6: Poloměr drona
    #             pursuers_obs[start+6] = new_purs_rads[idx] / MAX_DRONE_RAD    
    #             # --- 7: Povrchová vzdálenost kolegy ode MĚ ---
    #             # Výpočet reálné vzdálenosti (předtím byla norm_dists normalizovaná vůči vis_range)
    #             raw_mate_to_me_dist = np.linalg.norm(new_purs_pos[idx] - self.position)
    #             surf_mate_to_me_dist = max(0.0, raw_mate_to_me_dist - new_purs_rads[idx] - self.my_rad)
    #             # Uložíme normalizovanou vůči vis_range (aby to navazovalo na předchozí logiku)
    #             pursuers_obs[start+7] = surf_mate_to_me_dist / self.vis_range
    #             # --- 8: Povrchová vzdálenost kolegy od INVADERA ---
    #             raw_mate_to_inv_dist = np.linalg.norm(new_purs_pos[idx] - invader_pos)
    #             surf_mate_to_inv_dist = max(0.0, raw_mate_to_inv_dist - new_purs_rads[idx] - self.target["tar_rad"])
    #             # Uložíme normalizovanou vůči MAX_DIST
    #             pursuers_obs[start+8] = surf_mate_to_inv_dist / MAX_DIST
    #     density_obs = np.array([density / MAX_DENSITY], dtype=np.float32)
    #     # new_purs_pos = np.array(new_obs_pos)
    #     # new_purs_vel = np.array(new_obs_vel)
    #     # new_purs_rads = np.array(new_obs_rad)
    #     #pursuer state + density
    #     # pursuers_form = np.full(32, FAR_AWAY, dtype=np.float32)
    #     # for i in range(4):
    #     #     start = i * 8
    #     #     pursuers_form[start+3 : start+6] = 0.0
    #     #     pursuers_form[start+6] = 0.0
    #     # if len(new_purs_pos) > 0:
    #     #     density = len(new_purs_pos)
    #     #     norm_rel_positions = (new_purs_pos - self.position) / 2.0 #self.vis_range
    #     #     norm_dists = np.linalg.norm(norm_rel_positions, axis=1)
    #     #     closest_indices = np.argsort(norm_dists)[:4]
    #     #     #iterating from closest indices
    #     #     for i, idx in enumerate(closest_indices):
    #     #         start = i * 8  # <-- Nový multiplikátor 8!    
    #     #         pursuers_form[start : start+3] = norm_rel_positions[idx]    
    #     #         rel_vel = new_purs_vel[idx] - self.curr_speed
    #     #         pursuers_form[start+3 : start+6] = rel_vel / (MAX_SPEED * 2.0)    
    #     #         pursuers_form[start+6] = new_purs_rads[idx] / MAX_DRONE_RAD    
    #     #         pursuers_form[start+7] = norm_dists[idx]
    #     #invader state
    #     # Vektory (směry) zůstávají nezměněné, ukazují přesně na středy
    #     inv_rel_pos = (self.target["tar_pos"] - self.position) / MAX_DIST
    #     inv_rel_vel = self.target["tar_vel"] - self.curr_speed
    #     inv_to_prime_vec = (self.prime_pos - self.target["tar_pos"]) / MAX_DIST
    #     # --- NOVÉ: Povrchové vzdálenosti ---
    #     # 1. Vzdálenost k Prime dronovi (Invader -> Prime)
    #     raw_inv_prime_dist = np.linalg.norm(self.prime_pos - self.target["tar_pos"])
    #     surf_inv_prime_dist = max(0.0, raw_inv_prime_dist - self.target["tar_rad"] - self.prime_rad)
    #     inv_to_prime_dist = np.array([surf_inv_prime_dist / MAX_DIST], dtype=np.float32)
    #     # 2. Vzdálenost ke mně (Invader -> Pursuer)
    #     raw_inv_purs_dist = np.linalg.norm(self.target["tar_pos"] - self.position)
    #     surf_inv_purs_dist = max(0.0, raw_inv_purs_dist - self.target["tar_rad"] - self.my_rad)
    #     inv_purs_dist = np.array([surf_inv_purs_dist / MAX_DIST], dtype=np.float32)
    #     # --- Zbytek zůstává nezměněn ---
    #     inv_rad_obs = np.array([self.target["tar_rad"] / MAX_DRONE_RAD], dtype=np.float32)
    #     invaders_obs = np.concatenate([inv_rel_pos, inv_rel_vel / (MAX_SPEED * 2.0), inv_to_prime_vec, inv_to_prime_dist,
    #                    inv_rad_obs, inv_purs_dist])
    #     #closest obstacles
    #     obstacles_obs = np.full(20, FAR_AWAY, dtype=np.float32)
    #     #default radii
    #     obstacles_obs[3] = 0.0 
    #     obstacles_obs[8] = 0.0 
    #     obstacles_obs[13] = 0.0 
    #     obstacles_obs[18] = 0.0 
    #     obs_centers = self.obs_centers
    #     obs_radii = self.obs_radii
    #     if obs_centers is not None and len(obs_centers) > 0:
    #         #dist to centers
    #         center_dists = np.linalg.norm(obs_centers - self.position, axis=1)
    #         #dist to edges
    #         edge_dists = center_dists - obs_radii
    #         #only those in sight
    #         visible_mask = edge_dists <= self.vis_range
    #         visible_indices = np.where(visible_mask)[0]
    #         if len(visible_indices) > 0:
    #             #only visible distances
    #             visible_edge_dists = edge_dists[visible_indices]
    #             #sorting by distance
    #             sorted_local_indices = np.argsort(visible_edge_dists)[:4]
    #             closest_obs_indices = visible_indices[sorted_local_indices]
    #             #writing to the obs space
    #             for i, idx in enumerate(closest_obs_indices):
    #                 start = i * 5
    #                 #rel position
    #                 obs_pos = (obs_centers[idx] - self.position) / self.vis_range
    #                 obstacles_obs[start : start+3] = obs_pos
    #                 #radius
    #                 obstacles_obs[start+3] = obs_radii[idx] / MAX_RADIUS
    #                 #distance
    #                 obstacles_obs[start+4] = np.linalg.norm(obs_pos)
                    
    #     # --- VÝPOČET LOS (Line of Sight) PRO OBSERVATION ---
    #     attack_vector = self.target["tar_pos"] - self.prime_pos
    #     attack_dist = np.linalg.norm(attack_vector)
    #     if attack_dist > 0.1:
    #         prime_to_me = self.position - self.prime_pos
    #         # 1. Projekce (kde na ose jsem)
    #         # Normalizovat nepotřebujeme, už to dává hodnoty typicky kolem 0.0 až 1.0
    #         projection = np.dot(prime_to_me, attack_vector) / (attack_dist**2)
    #         # 2. Vzdálenost k ose
    #         cross_prod = np.cross(attack_vector, prime_to_me)
    #         raw_dist_to_line = np.linalg.norm(cross_prod) / attack_dist
    #         # 3. Vektor směrem k ose (Kudy tam?)
    #         # Najdeme ten nejbližší bod na ose:
    #         closest_point_on_line = self.prime_pos + (projection * attack_vector)
    #         # Vektor ode mě k tomu bodu:
    #         vector_to_line = closest_point_on_line - self.position    
    #     else:
    #         # Fallback, pokud by se nějak spawnuli na sobě
    #         projection = 0.0
    #         raw_dist_to_line = MAX_DIST
    #         vector_to_line = np.zeros(3)
    #     # Zabijácký LOS blok pro observation (5 hodnot)
    #     los_obs = np.concatenate([
    #         [projection],                                  # 1 hodnota
    #         [raw_dist_to_line / MAX_DIST],                 # 1 hodnota (normalizovaná!)
    #         vector_to_line / MAX_DIST                      # 3 hodnoty (vektor X, Y, Z)
    #     ]).astype(np.float32)
    #     #final vector
    #     final_obs = np.concatenate([
    #         my_obs,         # 10
    #         prime_obs,      # 8
    #         density_obs,    # 1
    #         pursuers_obs,   # 36
    #         #pursuers_form,  # 32
    #         invaders_obs,   # 12
    #         obstacles_obs,   # 20
    #         los_obs         # 5
    #     ]).astype(np.float32)
    #     return final_obs

    # def get_avoidance_direction(self, obstacle_pos, obstacle_rad, prime):
    #     #if prime is too close to obstacle, calculate which direction is better for avoidance, in 2D
    #     if np.linalg.norm(obstacle_pos - prime.position) - prime.my_rad - obstacle_rad <= self.obs_rad:
    #         if self.axis_found:
    #             return True
    #         vel = prime.curr_speed[:2]
    #         if np.linalg.norm(vel) < 0.1:
    #             return
    #         vec_to_obs = (obstacle_pos - prime.position)[:2]
    #         #2D cross product
    #         cross_z = vel[0] * vec_to_obs[1] - vel[1] * vec_to_obs[0]
    #         if cross_z > 0:
    #             self.axis_found = True
    #             self.circle_dir_obs = -1
    #         else:
    #             self.axis_found = True
    #             self.circle_dir_obs = 1
    #         return True
    #     return False

    # def calculate_prime_avoidance_axis(self, inv, prime_pos):
    #     vec_to_prime = prime_pos - inv.position
    #     #inv velocity vector
    #     velocity = inv.curr_speed
    #     if np.linalg.norm(velocity) < 0.1:
    #         velocity = np.array([0.0, 0.0, 1.0])
    #     #axis for avoidance
    #     avoidance_axis = np.cross(velocity, vec_to_prime)
    #     #if inv flies directly into the prime
    #     norm = np.linalg.norm(avoidance_axis)
    #     if norm < 1e-6:
    #         avoidance_axis = np.cross(velocity, np.array([0,0,1]))
    #         norm = np.linalg.norm(avoidance_axis)
    #     avoidance_axis = -avoidance_axis / norm
    #     return avoidance_axis

    # def calculate_obstacle_avoidance_axis(self, unit, obstacle_pos, obstacle_rad):
    #     #calculating if prime is too close to obstacle
    #     vec_to_obs = obstacle_pos - unit.position
    #     dist = np.linalg.norm(vec_to_obs) - unit.my_rad - obstacle_rad
    #     #prime velocity vector
    #     velocity = unit.curr_speed
    #     speed = np.linalg.norm(unit.curr_speed)
    #     v = np.zeros_like(self.position)
    #     if speed > 1e-6:
    #         v = velocity/speed
    #     vec_to_obs_norm = np.linalg.norm(vec_to_obs)
    #     vto_normed = np.zeros_like(self.position)
    #     if vec_to_obs_norm > 1e-6:
    #         vto_normed = vec_to_obs/vec_to_obs_norm
    #     if dist > self.obs_rad or np.dot(v, vto_normed) <= -0.2:
    #         self.avoid_axis = None
    #         return None
    #     elif self.avoid_axis is not None:
    #         return self.avoid_axis
    #     if np.linalg.norm(velocity) < 0.1:
    #         velocity = np.array([0.0, 0.0, 1.0])
    #     #axis for avoidance
    #     avoidance_axis = np.cross(velocity, vec_to_obs)
    #     #if prime flies directly into the obstacle
    #     norm = np.linalg.norm(avoidance_axis)
    #     if norm < 1e-6:
    #         avoidance_axis = np.cross(velocity, np.array([0,0,1]))
    #         norm = np.linalg.norm(avoidance_axis)
    #     avoidance_axis = -avoidance_axis / norm
    #     self.avoid_axis = avoidance_axis
    #     return self.avoid_axis

    # def get_nearest_obstacle(self, prime):
    #     #obstacle centers and radiuses
    #     obs_centers = self.obs_centers
    #     obs_radii = self.obs_radii
    #     #vector from prime to obstacle center
    #     vecs_to_obs = obs_centers - prime.position
    #     #distances center to center
    #     dists_center = np.linalg.norm(vecs_to_obs, axis=1)
    #     #distance from surface to surface
    #     dists_surface = dists_center - obs_radii - prime.my_rad
    #     #speed of prime
    #     speed_norm = np.linalg.norm(prime.curr_speed)
    #     #if prime is moving, interesting are only those obstacles in a way
    #     if speed_norm > 0.1:
    #         #only those who has positive dot product
    #         dot_products = np.sum(prime.curr_speed * vecs_to_obs, axis=1)
    #         mask = dot_products > 0
    #         #if there is none, not important
    #         if not np.any(mask):
    #             return None
    #         #relevant obstacles
    #         relevant_indices = np.where(mask)[0]
    #         relevant_dists = dists_surface[mask]
    #         #minimum
    #         min_idx_local = np.argmin(relevant_dists)
    #         real_idx = relevant_indices[min_idx_local]
    #         obstacle = {'center': self.obs_centers[real_idx], 'radius': self.obs_radii[real_idx]}
    #         return obstacle
    #     else:
    #         #just the closest
    #         min_idx = np.argmin(dists_surface)
    #         obstacle = {'center': self.obs_centers[min_idx], 'radius': self.obs_radii[min_idx]}
    #         return obstacle 

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