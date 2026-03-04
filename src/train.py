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

class UpdateSwarmCallback(BaseCallback):
    #callback for updating the brain
    def __init__(self, env, update_freq=100000, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.update_freq = update_freq
        self.next_update = 0
        #absolute paths
        self.save_dir = os.path.abspath("./models")
        os.makedirs(self.save_dir, exist_ok=True) 
        self.temp_path = os.path.join(self.save_dir, "temp_swarm_brain")
    def _on_training_start(self) -> None:
        #finding out the start step
        start_step = self.model.num_timesteps
        #computing next milestone
        zbytek = start_step % self.update_freq
        self.next_update = start_step + (self.update_freq - zbytek)
    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_update:
            print(f"\n[INFO] Step {self.num_timesteps}: updating brain!")
            self.model.save(self.temp_path)
            self.env.env_method("load_teammate_brain", self.temp_path)
            self.next_update += self.update_freq    
        return True

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
            inv_acc=new_inv_acc, purs_speed=[new_purs_speed], inv_speed=new_inv_speed, prime_speed=0.2, inv_pos=[np.array([10.0, 10.0, 10.0])], herding=True)
        env = HerdingEnv(world_instance=world, sc=new_sc)
        env = Monitor(env)
        return env
    return _init

def main_defense():
    #init
    _3d = True
    sc = Sim3DConfig(dt=0.02, purs_num=20, inv_num=1, obstacle=False, obstacle_rad=[3.0, 4.0], 
                     obstacle_pos=[np.array([13.0, 13.0, 6.0]), np.array([17.0, 6.0, 3.0])])
    inv_pos = np.array([[25.24, 20.15, 15.58]])
    world = SimulationWorld(sc, _3d=_3d, purs_acc=3.5, inv_pos=inv_pos, inv_acc=2.0, prime_acc=1.3, 
                            purs_speed=6.0, inv_speed=4.0, prime_speed=3.5)
    #gym
    env = FastWorldEnv(world_instance=world)
    #model
    print("Creating AI model...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_drone_tensorboard/")
    #train
    print("Starting training...")
    model.learn(total_timesteps=1_000_000)
    #saving result
    print("Training done...")
    model.save("drone_swarm_brain_gen1")
    
def main_herding():
    num_cpu = 6
    #vectorized env
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)], start_method="spawn")
    vec_env.env_method("load_teammate_brain", "drone_herding_brain_gen1.zip")
    #model
    custom_policy = dict(activation_fn=nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256]))
    print("Creating custom AI model...")
    #model = PPO("MlpPolicy", vec_env, policy_kwargs=custom_policy,verbose=1, 
    #    tensorboard_log="./ppo_drone_tensorboard/")
    custom_objects = {
        "ent_coef": 0.05,
        "learning_rate": 0.0003
    }
    model = PPO.load("drone_herding_brain_gen1", env=vec_env, custom_objects=custom_objects, tensorboard_log="./ppo_drone_tensorboard/", verbose=1)
    save_freq = 500000/num_cpu
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path='./models_checkpoints/',
        name_prefix='herding_brain')
    #train
    print("Starting training...")
    swarm_callback = UpdateSwarmCallback(vec_env, update_freq=100000)
    model.learn(total_timesteps=1_500_000, callback=swarm_callback, reset_num_timesteps=False, tb_log_name="PPO_Herd_Phase_2")
    #saving result
    print("Training done...")
    model.save("drone_herding_brain_gen2")

if __name__ == "__main__":
    main_herding()
    
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