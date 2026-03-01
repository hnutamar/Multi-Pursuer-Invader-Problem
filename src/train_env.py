import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from world import SimulationWorld
from sim_config3D import Sim3DConfig
from stable_baselines3 import PPO
from pursuer_states import States

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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(45,), dtype=np.float32)
        #episode limits
        self.max_steps = 1000
        self.current_step = 0
        if test:
            self.episode_num = 800000
        else:
            self.episode_num = 1
        #needed for reward
        self.last_inv_prime_dist = 0.0
        self.last_dist_to_inv = np.linalg.norm(self.world.pursuers[0].position - self.world.invaders[0].position)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #random max speeds
        new_purs_speed = random.uniform(6.0, 8.0)
        self.new_purs_acc = random.uniform(new_purs_speed/4, new_purs_speed/2)
        if 600000 > self.episode_num > 50000:
            new_inv_speed = max((6.0 / 600000) * self.episode_num, 0.5)
        elif 600000 < self.episode_num < 1000000:
            new_inv_speed = random.uniform(2.0, new_purs_speed + 2.0)
        elif self.episode_num > 1000000:
            new_inv_speed = random.uniform(2.0, new_purs_speed + 4.0)
        else:
            new_inv_speed = 0.5
        new_inv_acc = random.uniform(new_inv_speed/4, new_inv_speed/2)
        inv_pos = self.get_random_invader_start()
        purs_pos = self.get_random_pursuer_start()
        #brave new world
        self.world.reset(purs_acc=self.new_purs_acc, prime_acc=0.5, purs_pos=[purs_pos],
            inv_acc=new_inv_acc, purs_speed=new_purs_speed, inv_speed=new_inv_speed, prime_speed=1.0, inv_pos=[inv_pos])
        #setting to pursue state
        self.world.pursuers[0].state = States.PURSUE
        self.world.pursuers[0].target = {"target": self.world.invaders[0], "tar_pos": self.world.invaders[0].position, 
                           "tar_vel": self.world.invaders[0].curr_speed,"tar_rad": self.world.invaders[0].my_rad, "purs_type": 1}
        #first step
        self.world.step()
        obs = self.world.pursuers[0].get_observation_herding()
        #reseting steps
        self.current_step = 0
        self.last_inv_prime_dist = 0.0
        self.last_dist_to_inv = np.linalg.norm(self.world.pursuers[0].position - self.world.invaders[0].position)
        return obs, {}

    def get_random_invader_start(self):
        #random pos of invader, not too far or too close
        prime_pos = np.array([3.0, 3.0, 7.0])
        dist = np.random.uniform(15.0, 20.0)
        dir = np.random.randn(3)
        dir[2] = abs(dir[2]) 
        dir = dir / np.linalg.norm(dir)
        new_inv_pos = prime_pos + (dir * dist)
        new_inv_pos[2] = max(1.0, new_inv_pos[2])
        return new_inv_pos

    def get_random_pursuer_start(self):
        #random pos of pursuer, not too far or too close
        prime_pos = np.array([3.0, 3.0, 7.0])
        dist = np.random.uniform(1.5, 2.5)
        dir = np.random.randn(3)
        dir[2] = abs(dir[2]) 
        dir = dir / np.linalg.norm(dir)
        new_purs_pos = prime_pos + (dir * dist)
        new_purs_pos[2] = max(1.0, new_purs_pos[2])
        return new_purs_pos

    def step(self, action):
        self.episode_num += 1
        self.current_step += 1
        #getting action
        action = np.array(action, dtype=np.float32)
        #getting the true vector
        norm = np.linalg.norm(action)
        if norm > 1.0:
            action = action / norm
        target_acc = action * self.new_purs_acc
        #apply the new acc
        self.world.pursuers[0].herding(target_acc)
        #longer step, 0.02 is too small time step
        for _ in range(10): self.world.step()
        #computing reward
        prime = self.world.prime
        invader = self.world.invaders[0]
        pursuer = self.world.pursuers[0]
        #dist between prime nad invader
        current_inv_prime_dist = np.linalg.norm(invader.position - prime.position)
        #if episode is too long
        truncated = self.current_step >= self.max_steps
        reward = 0.0
        terminated = False
        #time penalization
        reward -= 1.0 
        curr_dist_to_inv = np.linalg.norm(pursuer.position - invader.position)
        #navigating penalty to invader
        distance_penalty = curr_dist_to_inv * 0.1 
        reward -= distance_penalty
        #penalty for pushing invader away (zero from same distance)
        SAFE_RADIUS = 25.0
        invader_threat_level = max(0.0, SAFE_RADIUS - current_inv_prime_dist)
        reward -= invader_threat_level * 0.5
        #updating for next step
        self.last_dist_to_inv = curr_dist_to_inv
        # #reward for pushing invader away
        # diff = current_inv_prime_dist - self.last_inv_prime_dist
        # #reward, positive if invader is further away from prime
        # reward += diff * 10.0  
        #if invader crashed, it is good, but better to push him away
        if invader.crashed:
            reward += 2.0
            terminated = True
        #penalization for crash
        if pursuer.crashed:
            reward -= 500.0
            terminated = True
        #penalization for breaking the defense
        elif current_inv_prime_dist < 3.0: 
            reward -= 500.0
            terminated = True
        #reward for getting invader far
        elif current_inv_prime_dist > 25.0:
            reward += 500.0
            terminated = True
        self.last_inv_prime_dist = current_inv_prime_dist    
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        #getting observation
        obs = self.world.pursuers[0].get_observation_herding()
        return obs