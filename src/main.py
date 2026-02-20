import time
from world import SimulationWorld
from visualizer import MatplotlibVisualizer
from sim_config3D import Sim3DConfig
from sim_config2D import Sim2DConfig
import numpy as np
import matplotlib.pyplot as plt

def main():
    _3d = True
    MANUAL_CONTROL = False
    #config
    if _3d:
        sc = Sim3DConfig(dt=0.02, purs_num=40, inv_num=5, obstacle=False, obstacle_rad=[3.0, 4.0], obstacle_pos=[np.array([13.0, 13.0, 6.0]), np.array([17.0, 6.0, 3.0])])
    else:
        sc = Sim2DConfig(dt=0.02, world_height=30, world_width=30, purs_num=20, inv_num=5, obstacle=False, 
                         obstacle_rad=[3.0, 4.0, 4.0], obstacle_pos=[np.array([13.0, 13.0]), np.array([17.0, 6.0]), np.array([6.0, 17.0])])
    #world, physics
    inv_pos = np.array([[20.24, 30.15, 25.58]])
    world = SimulationWorld(sc, _3d=_3d, purs_acc=6.0, inv_acc=7.7, prime_acc=4.3, purs_speed=12.0, inv_speed=11.0, prime_speed=5.5)
    #visualization
    SHOW_VISUALIZATION = True
    vis = None
    if SHOW_VISUALIZATION:
        vis = MatplotlibVisualizer(sc_config=sc, _3d=_3d, quiver=False)
    RENDER_EVERY = 5
    EPISODE_NUM = 5
    step_counter = 1
    current_episode = 1
    #loop of the simulator
    running = True
    while running:
        #manual control of invader
        manual_action = None
        if vis and MANUAL_CONTROL:
            manual_action = vis.manual_vel
        #physics step
        state, done = world.step(manual_invader_vel=manual_action)
        step_counter += 1
        #graphics
        if vis and step_counter % RENDER_EVERY == 0:
            if not vis.is_open:
                print("Simulation ends!")
                running = False
                break
            vis.render(state, world_instance=world)
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