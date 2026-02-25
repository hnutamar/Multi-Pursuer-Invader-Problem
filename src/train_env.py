import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from world import SimulationWorld
from sim_config3D import Sim3DConfig
from stable_baselines3 import PPO

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