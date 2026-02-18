import time
from world import SimulationWorld
from visualizer import MatplotlibVisualizer
from sim_config3D import Sim3DConfig
from sim_config2D import Sim2DConfig
import numpy as np
import matplotlib.pyplot as plt

def main():
    _3d = True
    #config
    if _3d:
        sc = Sim3DConfig(purs_num=30, inv_num=0, obstacle=False, obstacle_rad=[3.0, 4.0], obstacle_pos=[np.array([13.0, 13.0, 6.0]), np.array([17.0, 6.0, 3.0])])
    else:
        sc = Sim2DConfig(world_height=30, world_width=30, purs_num=10, inv_num=1, obstacle=True, 
                         obstacle_rad=[3.0, 4.0, 4.0], obstacle_pos=[np.array([13.0, 13.0]), np.array([17.0, 6.0]), np.array([6.0, 17.0])])
    #world, physics
    inv_pos = np.array([[20.24, 30.15, 25.58]])
    world = SimulationWorld(sc, _3d=_3d, purs_acc=2.8, inv_acc=1.7, prime_acc=0.3)
    #visualization
    SHOW_VISUALIZATION = True
    vis = None
    if SHOW_VISUALIZATION:
        vis = MatplotlibVisualizer(sc_config=sc, _3d=_3d)
    RENDER_EVERY = 2
    EPISODE_NUM = 2
    step_counter = 1
    current_episode = 1
    #loop (used also for RL training)
    running = True
    while running:
        #manual control of invader
        manual_action = None
        if vis:
            manual_action = vis.manual_vel
        #physics step
        state, done = world.step(dt=0.1)
        step_counter += 1
        #graphics
        if vis and step_counter % RENDER_EVERY == 0:
            if not vis.is_open:
                print("Simulation ends!")
                running = False
                break
            vis.render(state, world_instance=world, quiver=False)
            plt.pause(0.001)
        #end of episode check
        if done:
            print("End of episode!")
            if EPISODE_NUM == current_episode:
                running = False
                break
            world.reset()
            step_counter = 1
            current_episode += 1

if __name__ == "__main__":
    main()