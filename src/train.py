import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from stable_baselines3 import PPO
import time
from world import SimulationWorld
from train_env import FastWorldEnv
from train_env import HerdingEnv
from sim_config3D import Sim3DConfig
import numpy as np
import random
import torch.nn as nn
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from typing import Callable
import glob
import torch

class UpdateSwarmCallback(BaseCallback):
    def __init__(self, env, update_freq=500000, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.update_freq = update_freq
        self.next_update = 0
        self.save_dir = os.path.abspath("./models/history_def")
        os.makedirs(self.save_dir, exist_ok=True) 
        #gen1
        self.generation = 12
    def _on_training_start(self) -> None:
        start_step = self.model.num_timesteps
        rest = start_step % self.update_freq
        self.next_update = start_step + (self.update_freq - rest)
    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_update:
            self.generation += 1
            print(f"\n[INFO] Step {self.num_timesteps}: Making gen {self.generation}!")        
            #new generation
            new_brain_path = os.path.join(self.save_dir, f"gen_{self.generation}")
            self.model.save(new_brain_path)        
            #every brain in archive
            all_brains = glob.glob(os.path.join(self.save_dir, "gen_*.zip"))
            #sorting according to the time of creation
            all_brains.sort(key=os.path.getmtime)
            if all_brains:
                #newest model
                latest_brain = all_brains[-1].replace('.zip', '')
                #second newest
                if len(all_brains) > 1:
                    previous_brain = all_brains[-2].replace('.zip', '')
                else:
                    previous_brain = latest_brain
                #iterating through every env
                for env_idx in range(self.env.num_envs):
                    if self.env.num_envs // 2 > env_idx:
                        new_brain = latest_brain
                    else:
                        new_brain = previous_brain
                    print(f"[INFO] Env {env_idx} loading brain: {os.path.basename(new_brain)}")
                    self.env.env_method("load_teammate_brain", new_brain, indices=[env_idx])
            self.next_update += self.update_freq    
        return True
    
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def make_env(env_rank, seed=0):
    def _init():
        #unique seed
        set_random_seed(seed + env_rank)
        #new config
        new_sc = Sim3DConfig(dt=0.02, purs_num=1, inv_num=1, obstacle=False)
        #random max speeds
        new_purs_speed = random.uniform(2.0, 8.0)
        new_purs_acc = random.uniform(new_purs_speed/4, new_purs_speed/2)
        new_inv_speed = random.uniform(2.0, new_purs_speed + 2.0)
        new_inv_acc = random.uniform(new_inv_speed/4, new_inv_speed/2)
        #brave new world
        world = SimulationWorld(new_sc, _3d=True, purs_acc=[new_purs_acc], prime_acc=0.1, 
            inv_acc=[new_inv_acc], purs_speed=[new_purs_speed], inv_speed=[new_inv_speed], prime_speed=0.2, inv_pos=[np.array([10.0, 10.0, 10.0])], herding=True,
            no_target=True)
        env = HerdingEnv(world_instance=world, sc=new_sc)
        env = Monitor(env)
        return env
    return _init

def make_defense_env(env_rank, seed=0):
    def _init():
        #unique seed
        set_random_seed(seed + env_rank)
        #new config
        new_sc = Sim3DConfig(dt=0.02, purs_num=10, inv_num=1, obstacle=False)
        #random max speeds
        purs_acc = np.full(30, 5.0)
        inv_acc = np.full(30, 4.0)
        purs_speed = np.full(30, 8.0)
        inv_speed = np.full(30, 7.0)
        #brave new world
        model = PPO.load("./models/herding_modelC_0")
        model2 = PPO.load("./models/herding_modelC_0")
        world = SimulationWorld(new_sc, _3d=True, purs_acc=purs_acc, prime_acc=0.1, pursue_model=(model, model2),
            inv_acc=inv_acc, purs_speed=purs_speed, inv_speed=inv_speed, prime_speed=0.2, inv_pos=[np.array([10.0, 10.0, 10.0])], not_testing=True)
        env = FastWorldEnv(world_instance=world, sc=new_sc)
        env = Monitor(env)
        return env
    return _init

def main_defense():
    num_cpu = 8
    #vectorized env
    vec_env = SubprocVecEnv([make_defense_env(i) for i in range(num_cpu)], start_method="spawn")
    #model
    #custom_policy = dict(activation_fn=nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256]))
    print("Creating custom AI model...")
    #model = PPO("MlpPolicy", vec_env, policy_kwargs=custom_policy, batch_size=256, gamma=0.99, n_steps=2048,
    #  tensorboard_log="./ppo_drone_tensorboard/", learning_rate=linear_schedule(0.0003), verbose=1)
    #init_brain_path = "new_obs_best"
    init_brain_path = "./models/history_def/gen_12"
    init_brain_path2 = "./models/history_def/gen_11"
    #model.save(init_brain_path)
    vec_env.env_method("load_teammate_brain", init_brain_path, model_path2 = init_brain_path2)
    custom_objects = {
        #"ent_coef": 0.0001,
        #"learning_rate": 0.00005
    }
    model = PPO.load("./models/history_def/gen_12", env=vec_env, custom_objects=custom_objects, tensorboard_log="./ppo_drone_tensorboard/", verbose=1)
    # with torch.no_grad():
    #     model.policy.log_std.data = torch.full_like(model.policy.log_std.data, -1.5)
    #save_freq = 500000/num_cpu
    #checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path='./models_checkpoints/',
    #    name_prefix='herding_brain')
    #train
    print("Starting training...")
    swarm_callback = UpdateSwarmCallback(vec_env, update_freq=300_000)
    model.learn(total_timesteps=5_000_000, callback=swarm_callback, tb_log_name="PPO_Defense", reset_num_timesteps=False)
    #saving result
    print("Training done...")
    model.save("drone_defense_brain")
    
def main_herding():
    num_cpu = 10
    #vectorized env
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)], start_method="spawn")
    #model
    #custom_policy = dict(activation_fn=nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256]))
    print("Creating custom AI model...")
    #model = PPO("MlpPolicy", vec_env, policy_kwargs=custom_policy, verbose=1, 
    #    tensorboard_log="./ppo_drone_tensorboard/", learning_rate=linear_schedule(0.0003))
    #init_brain_path = "new_obs_best"
    init_brain_path = "./models/history/gen_40"
    init_brain_path2 = "./models/history/gen_40"
    #model.save(init_brain_path)
    vec_env.env_method("load_teammate_brain", init_brain_path, model_path2=init_brain_path2)
    custom_objects = {
        "ent_coef": 0.0001,
        "learning_rate": 0.00001
    }
    model = PPO.load("./models/history/gen_40", env=vec_env, custom_objects=custom_objects, tensorboard_log="./ppo_drone_tensorboard/", verbose=1)
    #with torch.no_grad():
    #    model.policy.log_std.data = torch.full_like(model.policy.log_std.data, -1.5)
    #save_freq = 500000/num_cpu
    #checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path='./models_checkpoints/',
    #    name_prefix='herding_brain')
    #train
    print("Starting training...")
    swarm_callback = UpdateSwarmCallback(vec_env, update_freq=300_000)
    model.learn(total_timesteps=20_000_000, callback=swarm_callback, tb_log_name="PPO_Marathon", reset_num_timesteps=False)
    #saving result
    print("Training done...")
    model.save("drone_herding_brain_gen2")

if __name__ == "__main__":
    main_defense()
    
#UNUSED CODE
    # #new config
    # new_sc = Sim3DConfig(dt=0.02, purs_num=1, inv_num=1, obstacle=False)
    # #random max speeds
    # new_purs_speed = random.uniform(2.0, 8.0)
    # new_purs_acc = random.uniform(new_purs_speed/4, new_purs_speed/2)
    # new_inv_speed = random.uniform(2.0, new_purs_speed + 2.0)
    # new_inv_acc = random.uniform(new_inv_speed/4, new_inv_speed/2)
    # #brave new world
    # world = SimulationWorld(new_sc, _3d=True, purs_acc=new_purs_acc, prime_acc=0.1, 
    #     inv_acc=new_inv_acc, purs_speed=new_purs_speed, inv_speed=new_inv_speed, prime_speed=0.2, inv_pos=[np.array([10.0, 10.0, 10.0])], herding=True)
    # env = HerdingEnv(world_instance=world, sc=new_sc)