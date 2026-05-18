import time
import matplotlib
matplotlib.use('TkAgg')
from world import SimulationWorld
from visualizer import MatplotlibVisualizer
from sim_config3D import Sim3DConfig
from sim_config2D import Sim2DConfig
import numpy as np
import matplotlib.pyplot as plt
from pybullet_visualizer import PyBulletVisualizer
from toronto_visualizer import TorontoVisualizer
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
import torch
import random

def lock_all_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def main():
    win_rates = []
    coll_rates = []
    mean_dists = []
    cap_rates = []
    kill_rates = []
    model_name = "Variant 1"
    seed_num = 640
    _3d = True
    PYBULLET = False
    MANUAL_CONTROL = False
    obses = [6, 6, 0, 0]
    INV_NUM = [1, 8, 1, 8]
    PUR_NUM = [5, 20, 5, 20]
    kamikadze = [True, True, True, True]
    #MODEL C
    model = PPO.load("./models/herding_modelC_0")
    model2 = PPO.load("./models/herding_modelC_0")
    #VARIANT 1
    #def_model = PPO.load("./models/def_final_restrictive")
    #VARIANT 2
    #def_model = PPO.load("./models/def_B_final")
    #HARDCODED
    def_model = None
    print("def model: " + str(def_model))
    for i in range(len(INV_NUM)):
        obstacles = obses[i]
        kam = kamikadze[i]
        lock_all_seeds(seed_num)
        #config
        if _3d:
            sc = Sim3DConfig(dt=0.02, purs_num=PUR_NUM[i], inv_num=INV_NUM[i], obstacle=obstacles, obstacle_rad=[3.0, 4.0], obstacle_pos=[np.array([13.0, 13.0, 6.0]), np.array([17.0, 6.0, 3.0])])
        else:
            sc = Sim2DConfig(dt=0.02, world_height=30, world_width=30, purs_num=20, inv_num=5, obstacle=True, 
                            obstacle_rad=[4.0, 4.0], obstacle_pos=[np.array([17.0, 6.0]), np.array([6.0, 17.0])])
        #world, physics
        inv_pos = np.array([[25.24, 20.15, 15.58]])
        world = SimulationWorld(sc, _3d=_3d, purs_acc=np.full(30, 3.0), inv_acc=np.full(30, 2.5), prime_acc=1.3, purs_speed=np.full(30, 7.0), inv_speed=np.full(30, 5.5), prime_speed=3.5, pursue_model=(model, model2), def_model=def_model, not_testing=True,
                                kamikadze=kam)
        #visualization
        SHOW_VISUALIZATION = False
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
            EPISODE_NUM = 200
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
                if current_episode % 25 == 0:
                    print("Episode: " + str(current_episode))
                if EPISODE_NUM == current_episode:
                    print("End of sim, " + str(world.episodes_won) + " won")
                    running = False
                    break
                lock_all_seeds(seed_num + current_episode)
                world.reset()
                if vis:
                    vis.reset()
                step_counter = 1
                current_episode += 1
        if vis and PYBULLET:
            vis.close()
        #data for plots    
        purs_purs_crash = world.purs_purs_coll
        purs_obs_crash = world.purs_obs_coll
        purs_prime_crash = world.purs_prime_coll
        purs_gr_crash = world.purs_gr_coll
        inv_prime = world.inv_prime_coll
        lost = world.prime_crash
        #normalized data
        win_rate = 1 - (lost / float(EPISODE_NUM))
        total_coll = purs_obs_crash + purs_prime_crash + purs_purs_crash + purs_gr_crash
        coll_rate = total_coll / float(EPISODE_NUM)
        cap_rate = world.capture_time
        #kill_rate = world.invaders_killed

        win_rates.append(win_rate)
        coll_rates.append(coll_rate)
        cap_rates.append(cap_rate)
        #kill_rates.append(kill_rate)
    #saving
    clean_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    filename_win = f"win_{clean_name}_rate_obs_{obstacles}_kamikadze_{kam}.npy"
    filename_coll = f"coll_{clean_name}_rate_obs_{obstacles}_kamikadze_{kam}.npy"
    filename_cap = f"cap_{clean_name}_rate_obs_{obstacles}_kamikadze_{kam}.npy"
    #filename_kill = f"kill_{clean_name}_rate_obs_{obstacles}_kamikadze_{kam}.npy"
    
    print("win rates: " + str(win_rates))
    print("coll rates: " + str(coll_rates))
    print("cap rates: " + str(cap_rates))
    #print("kill rates: " + str(kill_rates))
    #np array
    np.save(filename_win, np.array(win_rates))
    np.save(filename_coll, np.array(coll_rates))
    np.save(filename_cap, np.array(cap_rates))
    #np.save(filename_kill, np.array(kill_rates))
    #print(f"Data saved to {filename}")
    #plot_tracks(history_p, history_i, history_u)
    return

def plot_tracks(history_p, history_i, history_u):
    plt.ioff()
    print("Generating graph of trajectories")

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 12,
        "font.size": 11,
        "legend.fontsize": 10
    })

    # Set high DPI for crisp rendering in the thesis
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # 2. Remove the default grey background of 3D plot walls for a cleaner look
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    
    # Soften the grid lines
    ax.xaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 1)})
    ax.yaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 1)})
    ax.zaxis._axinfo["grid"].update({"color": (0.9, 0.9, 0.9, 1)})

    hist_p = np.array(history_p)
    hist_i = np.array(history_i)
    hist_u = np.array(history_u)

    # Helper function to extract X, Y, Z coordinates
    def get_xyz(arr_2d_or_3d):
        x = arr_2d_or_3d[:, 0]
        y = arr_2d_or_3d[:, 1]
        z = arr_2d_or_3d[:, 2] if arr_2d_or_3d.shape[1] >= 3 else np.full_like(x, 2.0)
        return x, y, z

    # 3. Pursuers (BLUE, representing the defensive swarm)
    for i in range(hist_p.shape[1]):
        x, y, z = get_xyz(hist_p[:, i, :])
        # Plot historical trajectory (dashed and slightly transparent)
        ax.plot(x, y, z, color='#1f77b4', alpha=0.5, linewidth=1.5, linestyle='--')
        # Plot current position marker (with a black edge for visibility)
        ax.scatter(x[-1], y[-1], z[-1], color='#1f77b4', s=60, marker='o', edgecolors='black', zorder=5)

    # 4. Invaders (RED, representing threats)
    if len(hist_i.shape) > 1 and hist_i.shape[1] > 0:
        for i in range(hist_i.shape[1]):
            x, y, z = get_xyz(hist_i[:, i, :])
            ax.plot(x, y, z, color='#d62728', alpha=0.5, linewidth=1.5)
            ax.scatter(x[-1], y[-1], z[-1], color='#d62728', s=60, marker='X', edgecolors='black', zorder=5)

    # 5. Prime (GREEN, the protected unit)
    x, y, z = get_xyz(hist_u)
    ax.plot(x, y, z, color='#2ca02c', linewidth=2.5, label='Prime Path')
    ax.scatter(x[-1], y[-1], z[-1], color='#2ca02c', s=80, marker='D', edgecolors='black', label='Prime Current', zorder=6)

    # Axis labels
    ax.set_xlabel(r'$X$ [m]', labelpad=10)
    ax.set_ylabel(r'$Y$ [m]', labelpad=10)
    ax.set_zlabel(r'$Z$ [m]', labelpad=10)
    
    # Legend
    ax.legend(loc='upper right')

    # 6. Fix the physical aspect ratio so the sphere formations aren't distorted
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    ax.set_box_aspect((x_range, y_range, z_range))

    # 7. Set initial camera angle for an isometric 3D view
    ax.view_init(elev=25, azim=-45)

    plt.tight_layout()
    fig.savefig("formation_diagram")
    plt.close()

if __name__ == "__main__":
    main()