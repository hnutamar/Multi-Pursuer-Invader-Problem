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
    world = SimulationWorld(new_sc, _3d=True, herding=True) 
    env = HerdingEnv(world_instance=world, sc=new_sc, test=True)
    #visualizer
    vis = MatplotlibVisualizer(sc_config=new_sc, _3d=True, quiver=False)
    #loading the model
    model_path = "drone_herding_brain_gen1" 
    print(f"Loading MLP: {model_path} ...")
    model = PPO.load(model_path)
    obs, info = env.reset()
    print("Start")
    #running forever
    running = True
    while running:
        #AI action
        action, _states = model.predict(obs, deterministic=True)
        #step
        obs, reward, terminated, truncated, info = env.step(action)
        world = env.world
        state = world.get_state()
        #controlling visualizer window
        if hasattr(vis, 'is_open') and not vis.is_open:
            print("Window closed, ending...")
            running = False
            break
        #rendering
        vis.render(state, world_instance=world)
        #restarting episode
        if terminated or truncated:
            print(f"Epizoda skončila! Poslední odměna: {reward:.1f}. Restartuji...")
            plt.pause(1.0)
            obs, info = env.reset()

if __name__ == "__main__":
    test_herding_model()