from stable_baselines3 import PPO
import time
from world import SimulationWorld
from train_env import FastWorldEnv
from sim_config3D import Sim3DConfig
import numpy as np

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

if __name__ == "__main__":
    main()