import time
from world import SimulationWorld
from visualizer import MatplotlibVisualizer
from sim_config3D import Sim3DConfig
from sim_config2D import Sim2DConfig
import numpy as np
import matplotlib.pyplot as plt
from pybullet_visualizer import PyBulletVisualizer
from toronto_visualizer import TorontoVisualizer
from mpl_toolkits.mplot3d import Axes3D

def main():
    _3d = True
    PYBULLET = False
    MANUAL_CONTROL = False
    #config
    if _3d:
        sc = Sim3DConfig(dt=0.02, purs_num=40, inv_num=0, obstacle=False, obstacle_rad=[3.0, 4.0], obstacle_pos=[np.array([13.0, 13.0, 6.0]), np.array([17.0, 6.0, 3.0])])
    else:
        sc = Sim2DConfig(dt=0.02, world_height=30, world_width=30, purs_num=5, inv_num=0, obstacle=False, 
                         obstacle_rad=[4.0, 4.0], obstacle_pos=[np.array([17.0, 6.0]), np.array([6.0, 17.0])])
    #world, physics
    inv_pos = np.array([[20.24, 30.15, 25.58]])
    world = SimulationWorld(sc, _3d=_3d, purs_acc=3.5, inv_acc=2.0, prime_acc=4.3, purs_speed=6.0, inv_speed=5.0, prime_speed=4.5)
    #visualization
    SHOW_VISUALIZATION = True
    vis = None
    if SHOW_VISUALIZATION:
        if PYBULLET:
            #vis = PyBulletVisualizer(sc_config=sc, _3d=_3d)
            initial_state = world.get_state()
            vis = TorontoVisualizer(sc_config=sc, _3d=_3d, init_state=initial_state)
        else:
            vis = MatplotlibVisualizer(sc_config=sc, _3d=_3d, quiver=False)
    if PYBULLET:
        RENDER_EVERY = 1
        EPISODE_NUM = 1
    else:
        RENDER_EVERY = 5
        EPISODE_NUM = 5
    step_counter = 1
    current_episode = 1
    SYNC_INTERVAL = 20
    sync_counter = 0
    #loop of the simulator
    running = True
    #memory for graph
    history_p = []
    history_i = []
    history_u = []
    while running:
        #manual control of invader
        manual_action = None
        if vis and MANUAL_CONTROL:
            manual_action = vis.manual_vel
        #physics step
        state, done = world.step(manual_invader_vel=manual_action)
        #for graph
        history_p.append(np.array(state['pursuers']))
        history_i.append(np.array(state['invaders']))
        history_u.append(np.array(state['prime']))
        step_counter += 1
        #graphics
        if vis and step_counter % RENDER_EVERY == 0:
            if PYBULLET:
                sync_counter += 1
                success, real_state = vis.render(state, world_instance=world)
                if not success or not vis.is_open:
                    print("Simulation ends!")
                    running = False
                    break
                #synchronizing reality with virtual world
                if sync_counter >= SYNC_INTERVAL:
                    world.synchronize_with_reality(real_state)
                    sync_counter = 0
            else:
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
    if vis and PYBULLET:
        vis.close()
        
    plt.ioff()
    #post-processing
    print("Generating graph of trajectories")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    #matrixes
    hist_p = np.array(history_p)
    hist_i = np.array(history_i)
    hist_u = np.array(history_u)
    #2D or 3D
    def get_xyz(arr_2d_or_3d):
        x = arr_2d_or_3d[:, 0]
        y = arr_2d_or_3d[:, 1]
        z = arr_2d_or_3d[:, 2] if arr_2d_or_3d.shape[1] >= 3 else np.full_like(x, 2.0)
        return x, y, z
    #red pursuers
    for i in range(hist_p.shape[1]):
        x, y, z = get_xyz(hist_p[:, i, :])
        ax.plot(x, y, z, color='red', alpha=0.5, linewidth=1.0)
    #blue invaders
    if hist_i.shape[1] > 0:
        for i in range(hist_i.shape[1]):
            x, y, z = get_xyz(hist_i[:, i, :])
            ax.plot(x, y, z, color='blue', alpha=0.5, linewidth=1.0)
    #green prime
    x, y, z = get_xyz(hist_u)
    ax.plot(x, y, z, color='green', linewidth=2.5, label='Prime Drone')
    #set legend
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Swarm Trajectories')
    ax.legend()
    #fixing the aspect ration
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    #ranges
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    #right physical aspect
    ax.set_box_aspect((x_range, y_range, z_range))
    plt.show()

if __name__ == "__main__":
    main()