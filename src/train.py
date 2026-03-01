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

def main():
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
    #init
    _3d = True
    #new config
    new_sc = Sim3DConfig(dt=0.02, purs_num=1, inv_num=1, obstacle=False)
    #random max speeds
    new_purs_speed = random.uniform(2.0, 8.0)
    new_purs_acc = random.uniform(new_purs_speed/4, new_purs_speed/2)
    new_inv_speed = random.uniform(2.0, new_purs_speed + 2.0)
    new_inv_acc = random.uniform(new_inv_speed/4, new_inv_speed/2)
    #brave new world
    world = SimulationWorld(new_sc, _3d=True, purs_acc=new_purs_acc, prime_acc=0.5, 
        inv_acc=new_inv_acc, purs_speed=new_purs_speed, inv_speed=new_inv_speed, prime_speed=1.0, inv_pos=[np.array([10.0, 10.0, 10.0])], herding=True)
    #gym
    env = HerdingEnv(world_instance=world, sc=new_sc)
    #model
    custom_policy = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    print("Creating custom AI model...")
    model = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=custom_policy,
        verbose=1, 
        tensorboard_log="./ppo_drone_tensorboard/"
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=500000, 
        save_path='./models_checkpoints/',
        name_prefix='herding_brain'
    )
    #train
    print("Starting training...")
    model.learn(total_timesteps=2_000_000, callback=checkpoint_callback)
    #saving result
    print("Training done...")
    model.save("drone_herding_brain_gen1")

if __name__ == "__main__":
    main_herding()