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

def test_herding_model():
    print("Creating world for testing...")
    #setting env
    new_sc = Sim3DConfig(dt=0.02, purs_num=1, inv_num=1, obstacle=False)
    world = SimulationWorld(new_sc, _3d=True, herding=True, purs_speed=[4.0], purs_acc=[3.0], no_target=True) 
    env = HerdingEnv(world_instance=world, sc=new_sc, test=True)
    #loading the model
    #model_path = "./models_checkpoints/herding_brain_1000000_steps" 
    #model_path = "brain_to_integrate" 
    model_path = "./models/history/gen_12" 
    print(f"Loading MLP: {model_path} ...")
    model = PPO.load(model_path)
    env.load_teammate_brain(model_path)
    obs, info = env.reset()
    print("Start")
    #running forever
    running = True
    whole_reward = 0
    episode_num = 0
    render_every = 1
    ep_len = 0
    #visualizer
    #vis = MatplotlibVisualizer(sc_config=env.sc, _3d=True, quiver=False)
    while running:
        #AI action
        ep_len += 1
        action, _states = model.predict(obs, deterministic=True)
        #step
        obs, reward, terminated, truncated, info = env.step(action)
        whole_reward += reward
        world = env.world
        state = world.get_state()
        # #controlling visualizer window
        # if hasattr(vis, 'is_open') and not vis.is_open:
        #     print("Window closed, ending...")
        #     running = False
        #     break
        # #rendering
        # if ep_len % render_every == 0:
        #     vis.render(state, world_instance=world)
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
            #plt.pause(1.0)
            obs, info = env.reset()
            # if hasattr(vis, 'is_open') and not vis.is_open:
            #    vis.is_open = False
            # #visualizer
            # vis = MatplotlibVisualizer(sc_config=env.sc, _3d=True, quiver=False)
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
    test_herding_model()