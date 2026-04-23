import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from world import SimulationWorld
from visualizer import MatplotlibVisualizer
from sim_config3D import Sim3DConfig
from train_env import HerdingEnv
from train_env import FastWorldEnv
import random
import torch

def lock_all_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def test_defense_model():
    print("Creating world for testing...")
    #setting env
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
    #loading the model
    #model_path = "./models_checkpoints/herding_brain_1000000_steps" 
    #model_path = "new_obs_best" 
    model_path = "./models/history_def/gen_9" 
    #model_path = "./models/gen_31" 
    #model_path2 = "./models/gen_30" 
    #model_path2 = "new_obs_best2"
    print(f"Loading MLP: {model_path} ...")
    #print(f"Loading MLP: {model_path2} ...")
    model = PPO.load(model_path)
    env.load_teammate_brain(model_path, model_path2=None)
    obs, info = env.reset()
    print("Start")
    #running forever
    running = True
    whole_reward = 0
    episode_num = 0
    render_every = 1
    ep_len = 0
    #visualizer
    vis = MatplotlibVisualizer(sc_config=env.sc, _3d=True, quiver=False)
    while running:
        #AI action
        ep_len += 1
        action, _states = model.predict(obs, deterministic=True)
        # if action[13] < -0.33:
        #     print("jej")
        #step
        obs, reward, terminated, truncated, info = env.step(action)
        whole_reward += reward
        world = env.world
        state = world.get_state()
        #controlling visualizer window
        if hasattr(vis, 'is_open') and not vis.is_open:
            print("Window closed, ending...")
            running = False
            break
        #rendering
        if ep_len % render_every == 0:
            vis.render(state, world_instance=world)
        #restarting episode
        if terminated or truncated:
            #print(f"Episode over! Reward: {whole_reward:.1f}.")
            whole_reward = 0
            ep_len = 0
            episode_num += 1
            if episode_num % 25 == 0:
                print("Episode: " + str(episode_num))
            if episode_num == 100:
                break
            plt.pause(1.0)
            obs, info = env.reset()
            if hasattr(vis, 'is_open') and not vis.is_open:
               vis.is_open = False
            #visualizer
            vis = MatplotlibVisualizer(sc_config=env.sc, _3d=True, quiver=False)
    purs_crash = env.lost_purs_crash
    prime_purs = env.lost_pursuer_prime
    inv_prime = env.lost_invader_prime
    print("Invader won: " + str(inv_prime))
    print("Pursuer crashed to Prime: " + str(prime_purs))
    print("Pursuer crashed somewhere: " + str(purs_crash))
    print("Pursuers killed everyone: " + str(env.all_invaders_dead))
    lost = prime_purs + inv_prime
    win_rate = 1 - (lost / float(episode_num))
    print("win rate:" + str(win_rate))

def test_herding_model():
    seed_num = 42
    lock_all_seeds(seed_num)
    print("Creating world for testing...")
    #setting env
    new_sc = Sim3DConfig(dt=0.02, purs_num=1, inv_num=1, obstacle=False)
    new_purs_speed = random.uniform(2.0, 8.0)
    new_purs_acc = random.uniform(new_purs_speed/4, new_purs_speed/2)
    new_inv_speed = random.uniform(2.0, new_purs_speed + 2.0)
    new_inv_acc = random.uniform(new_inv_speed/4, new_inv_speed/2)
    #brave new world
    world = SimulationWorld(new_sc, _3d=True, purs_acc=[new_purs_acc], prime_acc=0.1, 
        inv_acc=[new_inv_acc], purs_speed=[new_purs_speed], inv_speed=[new_inv_speed], prime_speed=0.2, inv_pos=[np.array([10.0, 10.0, 10.0])], herding=True,
        no_target=True)
    env = HerdingEnv(world_instance=world, sc=new_sc)
    #loading the model
    #model_path = "./models_checkpoints/herding_brain_1000000_steps" 
    #model_path = "brain_to_integrate" 
    model_path = "./models/history/gen_35" 
    #model_path = "./models/herding_modelB_2.0" 
    model_path2 = "./models/history/gen_35" 
    #model_path2 = "./models/herding_modelB_2.1" 
    #model_path2 = "brain_to_integrate" 
    print(f"Loading MLP: {model_path} ...")
    print(f"Loading MLP: {model_path2} ...")
    model = PPO.load(model_path)
    env.load_teammate_brain(model_path, model_path2=model_path2)
    obs, info = env.reset()
    print("Start")
    #running forever
    running = True
    whole_reward = 0 
    episode_num = 0
    render_every = 4
    ep_len = 0
    #visualizer
    vis = MatplotlibVisualizer(sc_config=env.sc, _3d=True, quiver=False)
    while running:
        #AI action
        ep_len += 1
        action, _states = model.predict(obs, deterministic=True)
        #step
        obs, reward, terminated, truncated, info = env.step(action)
        whole_reward += reward
        world = env.world
        state = world.get_state()
        #controlling visualizer window
        if hasattr(vis, 'is_open') and not vis.is_open:
            print("Window closed, ending...")
            running = False
            break
        #rendering
        if ep_len % render_every == 0:
            vis.render(state, world_instance=world)
        #restarting episode
        if terminated or truncated:
            #print(f"Episode over! Reward: {whole_reward:.1f}.")
            whole_reward = 0
            ep_len = 0
            episode_num += 1
            lock_all_seeds(seed_num + episode_num)
            if episode_num % 25 == 0:
                print("Episode: " + str(episode_num))
            if episode_num == 50:
                break
            plt.pause(1.0)
            obs, info = env.reset()
            if hasattr(vis, 'is_open') and not vis.is_open:
               vis.is_open = False
            #visualizer
            vis = MatplotlibVisualizer(sc_config=env.sc, _3d=True, quiver=False)
    purs_crash = env.lost_purs_crash
    prime_purs = env.lost_pursuer_prime
    inv_prime = env.lost_invader_prime
    print("Invader won: " + str(inv_prime))
    print("Pursuer crashed to Prime: " + str(prime_purs))
    print("Pursuer crashed somewhere: " + str(purs_crash))
    lost = prime_purs + inv_prime
    win_rate = 1 - (lost / float(episode_num))
    print("win rate:" + str(win_rate))

if __name__ == "__main__":
    test_defense_model()