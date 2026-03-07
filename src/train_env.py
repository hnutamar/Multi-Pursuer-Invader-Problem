import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from world import SimulationWorld
from sim_config3D import Sim3DConfig
from stable_baselines3 import PPO
from pursuer_states import States
import torch

class FastWorldEnv(gym.Env):
    def __init__(self, world_instance: SimulationWorld, previous_model_path=None):
        super().__init__()
        self.world = world_instance
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(43,), dtype=np.float32)
        #companion model
        self.companion_model = None
        if previous_model_path:
            self.companion_model = PPO.load(previous_model_path)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #randomization, pursuer num
        new_purs_num = random.randint(5, 40)
        new_inv_num = random.randint(1, new_purs_num)
        #random max speeds
        new_inv_speed = random.uniform(2.0, 6.0)
        #random obstacles
        num_obstacles = random.randint(0, 3)
        obs_rads = [random.uniform(2.0, 6.0) for _ in range(num_obstacles)]
        #random obstacle positioning
        obs_pos = [np.array([random.uniform(5, 25), random.uniform(5, 25), random.uniform(2, 8)]) for _ in range(num_obstacles)]
        #new config
        new_sc = Sim3DConfig(dt=0.02, purs_num=new_purs_num, inv_num=new_inv_num, 
            obstacle=(num_obstacles > 0), obstacle_rad=obs_rads, obstacle_pos=obs_pos)
        #random invader positioning
        new_inv_pos = np.array([[random.uniform(10, 20), random.uniform(10, 20), random.uniform(5, 15)] for _ in range(new_inv_num)])
        #brave new world
        self.world = SimulationWorld(new_sc, _3d=True, purs_acc=3.5, inv_pos=new_inv_pos, 
            inv_acc=2.0, purs_speed=6.0, inv_speed=new_inv_speed, prime_speed=3.5)
        #first step
        self.world.step()
        obs = self.world.pursuers[0].get_observation()
        return obs, {}

    def step(self, action):
        terminated = False
        all_invaders = self.world.free_invaders
        #first pursuer is learning
        vis_inv_0 = self.world.pursuers[0].get_closest_invaders(all_invaders, 2)
        self.world.pursuers[0].set_rl_action(action, vis_inv_0)
        #other pursuers
        for i in range(1, len(self.world.pursuers)):
            if self.companion_model is not None:
                #AI from prev generation
                obs_i = self.world.pursuers[i].get_observation()
                #getting action
                action_i, _ = self.companion_model.predict(obs_i, deterministic=True)
                vis_inv_i = self.world.pursuers[i].get_closest_invaders(all_invaders, 2)
                self.world.pursuers[i].set_rl_action(action_i, vis_inv_i)
            else:
                #not having prev model
                self.world.pursuers[i].is_rl_controlled = False
        MACRO_STEPS = 20
        #new rewards
        for _ in range(MACRO_STEPS):
            terminated = self.world.step()
            if terminated:
                break    
        #if pursuer crashed, stop episode
        if not self.world.pursuers[0].crashed:
            terminated = True
        #reward, observation
        obs = self.world.pursuers[0].get_observation()
        reward = self._compute_reward()
        return obs, reward, terminated, False, {}

class HerdingEnv(gym.Env):
    def __init__(self, world_instance: SimulationWorld, sc, test=False):
        super().__init__()
        self.world = world_instance
        self.sc = sc
        #action space - acc vector
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        #obs space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(65,), dtype=np.float32)
        #episode limits
        self.current_step = 0
        if test:
            self.episode_num = 1#np.inf
            self.max_steps = 500
        else:
            self.episode_num = 1
            self.max_steps = 1000
        #needed for reward
        self.last_inv_prime_dist = np.linalg.norm(self.world.prime.position - self.world.invaders[0].position)
        self.last_dist_to_inv = np.linalg.norm(self.world.pursuers[0].position - self.world.invaders[0].position)
        self.last_inv_pos = self.world.invaders[0].position
        self.obs_centers = []
        self.obs_rads = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #num of purs in episode
        new_purs_num = np.random.randint(1, 11)
        #pursuers
        new_purs_speeds = np.random.uniform(4.0, 8.0, size=new_purs_num)
        self.new_purs_accs = np.random.uniform(low=new_purs_speeds / 2.0, high=new_purs_speeds / 1.3)
        max_purs_speed = np.max(new_purs_speeds)
        #invaders
        new_inv_speed = np.random.uniform(2.0, max_purs_speed * 0.85)
        new_inv_acc = np.random.uniform(new_inv_speed / 2.0, new_inv_speed / 1.3)
        #prime
        new_prime_speed = 1.0
        new_prime_acc = np.random.uniform(new_prime_speed / 4.0, new_prime_speed / 2.0)
        #positions
        inv_pos = self.get_random_invader_start()
        purs_positions = self.get_random_pursuer_starts(new_purs_num)
        #random radii
        drone_rad = random.uniform(0.1, 0.4)
        prime_rad = random.uniform(0.1, 0.7)
        #update config
        self.world.sc.PURSUER_NUM = new_purs_num
        self.world.sc.DRONE_RAD = drone_rad
        self.world.sc.UNIT_RAD = prime_rad
        #obstacles
        num_obs = random.choice([0, 1, 2, 3, 4, 5])
        if num_obs > 0:
            #array of pos and radii
            all_agents_pos = np.vstack([inv_pos, purs_positions,np.array([3.0, 3.0, 7.0])])
            all_agents_rad = np.concatenate([[drone_rad],np.full(new_purs_num, drone_rad), [prime_rad]])
            #obs centers and radii
            self.obs_centers, self.obs_rads = self.generate_safe_obstacles(num_obs, all_agents_pos, all_agents_rad, 20, True)
            self.world.sc.obs_pos = self.obs_centers
            self.world.sc.obs_rads = self.obs_rads
            self.world.sc.obstacle = True
        else:
            self.world.sc.obstacle = False
            self.obs_centers = []
            self.obs_rads = []
        #brave new world
        self.world.reset(purs_acc=self.new_purs_accs, prime_acc=new_prime_acc, purs_pos=purs_positions,
            inv_acc=new_inv_acc, purs_speed=new_purs_speeds, inv_speed=new_inv_speed, prime_speed=new_prime_speed, inv_pos=[inv_pos])
        #setting to pursue state
        for i in range(new_purs_num):
            self.world.pursuers[i].state = States.PURSUE
            self.world.pursuers[i].target = {"target": self.world.invaders[0], "tar_pos": self.world.invaders[0].position, 
                           "tar_vel": self.world.invaders[0].curr_speed,"tar_rad": self.world.invaders[0].my_rad, "purs_type": 1}
        #first step
        self.world.step()
        obs = self.world.pursuers[0].get_observation_herding()
        #reseting steps
        self.current_step = 0
        self.last_inv_prime_dist = np.linalg.norm(self.world.prime.position - self.world.invaders[0].position)
        self.last_dist_to_inv = np.linalg.norm(self.world.pursuers[0].position - self.world.invaders[0].position)
        self.last_inv_pos = self.world.invaders[0].position
        return obs, {}
    
    def load_teammate_brain(self, model_path):
        #updating the brain
        self.teammate_brain = PPO.load(model_path, device="cpu")
        with torch.no_grad():
            self.teammate_brain.policy.log_std.data = torch.full_like(self.teammate_brain.policy.log_std.data, -2.1)

    def generate_safe_obstacles(self, num_obs, agent_positions, agent_radii, max_coord, is_3d, min_r=1.0, max_r=5.0, safe_margin=1.5):
        #arrays
        centers = []
        radii = []
        for _ in range(num_obs):
            placed = False
            #searching only 100 times, otherwise map could be full
            for _ in range(100): 
                #random rad
                r = np.random.uniform(min_r, max_r)
                #random pos
                if is_3d:
                    c = np.random.uniform(-max_coord, max_coord, size=3)
                    c[2] = np.random.uniform(r, max_coord) 
                else:
                    c = np.random.uniform(-max_coord, max_coord, size=2)
                #collision check
                dists = np.linalg.norm(agent_positions - c, axis=1)
                #with safe margin
                safe_dists = r + agent_radii + safe_margin
                #everything safe
                if not np.any(dists < safe_dists):
                    centers.append(c)
                    radii.append(r)
                    placed = True
                    break
        return np.array(centers), np.array(radii)

    def get_random_invader_start(self):
        #random pos of invader, not too far or too close
        prime_pos = np.array([3.0, 3.0, 7.0])
        dist = np.random.uniform(12.0, 17.0)
        dir = np.random.randn(3)
        dir[2] = abs(dir[2]) 
        dir = dir / np.linalg.norm(dir)
        new_inv_pos = prime_pos + (dir * dist)
        new_inv_pos[2] = max(1.0, new_inv_pos[2])
        return new_inv_pos

    def get_random_pursuer_starts(self, num_pursuers):
        #center, prime position
        prime_pos = np.array([3.0, 3.0, 7.0])    
        #array for all pursuers
        positions = np.zeros((num_pursuers, 3))
        #guard, the one close to prime
        dist_close = np.random.uniform(2.2, 3.0)
        dir_close = np.random.randn(3)
        dir_close[2] = abs(dir_close[2])
        dir_close = dir_close / np.linalg.norm(dir_close)
        positions[0] = prime_pos + (dir_close * dist_close)
        #others can be further
        if num_pursuers > 1:
            num_far = num_pursuers - 1
            #dist from prime
            dists_far = np.random.uniform(4.0, 15.0, size=num_far)    
            #random directions
            dirs_far = np.random.randn(num_far, 3)
            dirs_far[:, 2] = np.abs(dirs_far[:, 2])   
            #normalization
            norms = np.linalg.norm(dirs_far, axis=1, keepdims=True)
            dirs_far = dirs_far / norms    
            #computing final pos
            positions[1:] = prime_pos + (dirs_far * dists_far[:, np.newaxis])    
        #safe clip
        positions[:, 2] = np.clip(positions[:, 2], 1.0, np.inf)
        #so that learning pursuer is not always by prime
        rnd_num = np.random.randint(0, 4)
        if rnd_num != 1:
            np.random.shuffle(positions)
        return positions

    def step(self, action):
        self.episode_num += 1
        self.current_step += 1
        all_raw_actions = [action]
        #generating action of others
        if hasattr(self, 'teammate_brain') and self.teammate_brain is not None:
            for i in range(1, len(self.world.pursuers)):
                obs_i = self.world.pursuers[i].get_observation_herding() 
                action_i, _ = self.teammate_brain.predict(obs_i, deterministic=True)
                all_raw_actions.append(action_i)
        else:
            #does nothing, if there is no brain
            for i in range(1, len(self.world.pursuers)):
                all_raw_actions.append(np.zeros_like(action))
        #drone moving
        for i, pursuer in enumerate(self.world.pursuers):
            raw_act = np.array(all_raw_actions[i], dtype=np.float32)
            #cliping
            norm = np.linalg.norm(raw_act)
            if norm > 1.0:
                raw_act = raw_act / norm
            target_acc = raw_act * self.new_purs_accs[i]
            #herding
            pursuer.herding(target_acc)
        #frame skipping, 0.02 is too short
        self.world.step()    
        self.world.step()    
        self.world.step()    
        self.world.step()      
        state, done = self.world.step()
        #computing reward
        prime_pos = state["prime"]
        invader_pos = state["invaders"][0]
        invader_crashed = state["invaders_status"][0]
        pursuer_pos = state["pursuers"][0]
        pursuer_rad = self.world.pursuers[0].my_rad
        #dist between prime nad invader
        current_inv_prime_dist = np.linalg.norm(invader_pos - prime_pos)
        #if episode is too long
        truncated = self.current_step >= self.max_steps
        reward = 0.0
        reward += 0.1
        terminated = False
        curr_dist_to_inv = np.linalg.norm(pursuer_pos - invader_pos) - 5.0
        #navigating penalty to invader
        if curr_dist_to_inv > 0:
            distance_penalty = curr_dist_to_inv * 0.03
            reward -= distance_penalty
        pursuer_positions = np.array([p.position for p in self.world.free_purs])
        #pursuer penalty
        colleague_penalty = 0.0
        safe_drone_dist = 3.0
        other_rads = np.array([p.my_rad for p in self.world.free_purs[1:]])
        other_pos = pursuer_positions[1:]
        if len(other_pos) > 0:
            distances = np.linalg.norm(other_pos - pursuer_pos, axis=1) - pursuer_rad - other_rads
            violations = safe_drone_dist - distances
            colleague_penalty = -np.sum(violations[violations > 0]) * 0.1
            reward += colleague_penalty
        #obstacle penalty
        if len(self.obs_centers) > 0:
            obs_centers_arr = self.obs_centers
            obs_rads_arr = self.obs_rads
            obs_distances = np.linalg.norm(obs_centers_arr - pursuer_pos, axis=1) - pursuer_rad - obs_rads_arr
            obs_violations = safe_drone_dist - obs_distances
            obs_penalty = -np.sum(obs_violations[obs_violations > 0]) * 0.1
            reward += obs_penalty
        #ground penalty
        ground_dist = pursuer_pos[2] - pursuer_rad
        if ground_dist < safe_drone_dist:
            ground_penalty = (safe_drone_dist - ground_dist) * 0.1
            reward -= ground_penalty
        #prime penalty
        dist_to_prime = np.linalg.norm(pursuer_pos - self.world.prime.position) - pursuer_rad - self.world.prime.my_rad
        if dist_to_prime < safe_drone_dist:
            prime_violation = safe_drone_dist - dist_to_prime
            reward -= prime_violation * 0.2
        #COM reward
        if len(pursuer_positions) > 1:
            center_of_mass = np.mean(pursuer_positions, axis=0)
            invader_com_dist = np.linalg.norm(center_of_mass - invader_pos)
            com_reward = max(0.0, 5.0 - invader_com_dist) * 0.1
            reward += com_reward
        # #reward for pushing invader away
        diff = current_inv_prime_dist - self.last_inv_prime_dist
        #reward, positive if invader is further away from prime
        if current_inv_prime_dist < 20.0: # and diff > 0:
            reward += diff * 0.1
        #if invader crashed, it is good, but better to push him away
        if invader_crashed:
            reward += 10.0
            terminated = True
        #penalization for crash
        if self.world.pursuers[0].crashed:
            print("lost")
            reward -= 20.0
            terminated = True
        #penalization for breaking the defense
        if current_inv_prime_dist < 2.0 or done: 
            print("lost")
            reward -= 20.0
            terminated = True
        #reward for getting invader far
        safe_distance = min(current_inv_prime_dist, 20.0)
        safety_ratio = safe_distance / 20.0
        reward += safety_ratio * 0.05
        #penalty for invader moving too much
        # if current_inv_prime_dist > 20:
        #     inv_diff = np.linalg.norm(self.last_inv_pos - invader_pos)
        #     reward -= inv_diff * 0.05
        # self.last_inv_pos = invader_pos
        # elif current_inv_prime_dist > 20.0:
        #     reward += 0.5
        #whole game won
        if truncated:
            reward += min(current_inv_prime_dist, 20.0)
        self.last_inv_prime_dist = current_inv_prime_dist    
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        #getting observation
        obs = self.world.pursuers[0].get_observation_herding()
        return obs
    
#UNUSED CODE:
        #phase one, invader is basically on a spot
        # if self.episode_num <= 600000:
        #     new_inv_speed = random.uniform(1.0, 2.0)
        #     new_prime_speed = 0.05
        # #phase two, invader is slowly moving
        # elif 600000 < self.episode_num <= 1300000:
        #     new_inv_speed = random.uniform(2.0, (new_purs_speed / 1300000) * self.episode_num)
        #     new_prime_speed = max((1.0 / 1300000) * self.episode_num, 0.05)
        #phase three, small change in speed
        # elif 800000 < self.episode_num <= 1300000:
        #     progress = (self.episode_num - 800000) / 500000.0    
        #     max_allowed_speed = 2.0 + progress * (new_purs_speed - 1.0 - 1.0)    
        #     new_inv_speed = random.uniform(2.0, max_allowed_speed)
        #     new_prime_speed = 1.0
        #phase four, hardcore
        #else:
        
        #gass leak
        #action_penalty = np.sum(np.square(action)) * 0.01 
        #reward -= action_penalty
        #time penalization
        #reward -= 0.1 
        #curr_dist_to_inv = np.linalg.norm(pursuer_pos - invader_pos)
        #navigating penalty to invader
        #distance_penalty = curr_dist_to_inv * 0.01
        #reward -= distance_penalty
        #penalty for pushing invader away (zero from certain distance)
        # SAFE_RADIUS = 20.0
        # invader_threat_level = max(0.0, SAFE_RADIUS - current_inv_prime_dist)
        # reward -= invader_threat_level * 0.2
        #updating for next step
        #self.last_dist_to_inv = curr_dist_to_inv