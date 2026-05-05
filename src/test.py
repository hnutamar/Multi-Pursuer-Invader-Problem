import time
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from world import SimulationWorld
from visualizer import MatplotlibVisualizer
from sim_config3D import Sim3DConfig
from train_env import HerdingEnv
from train_env import FastWorldEnv
import random
# Čistý, minimalistický základ
plt.style.use('default')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['figure.dpi'] = 300
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

def plot_failure_histogram(failure_steps, max_steps=250, title="Distribution of Failure Steps"):
    if len(failure_steps) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(0, max_steps + 10, 10)
    
    # Jemná zelenkavá barva
    main_color = '#93C47D'
    
    # Tenký bílý okraj sloupců
    ax.hist(failure_steps, bins=bins, color=main_color, edgecolor='white', linewidth=1.0, alpha=0.8, zorder=3)
    
    mean_fail = np.mean(failure_steps)
    # Tenká, světle šedá přerušovaná čára pro průměr
    ax.axvline(mean_fail, color='#999999', linestyle='--', linewidth=1.5, 
                label=f'Mean Failure Step: {mean_fail:.1f}', zorder=4)
    
    # Odstranění zbytečných rámečků
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    
    ax.grid(axis='y', linestyle='-', alpha=0.4, color='#EEEEEE', zorder=0)
    
    ax.set_title(title, color='#333333', pad=15)
    ax.set_xlabel("Simulation Step When Invader Crashed into Prime", color='#555555', labelpad=10)
    ax.set_ylabel("Number of Episodes", color='#555555', labelpad=10)
    
    ax.legend(frameon=False, labelcolor='#555555')
    
    fig.tight_layout()
    plt.savefig("failure_histogram.png", bbox_inches='tight')
    # plt.show()

def test_herding_model():
    win_rates = []
    coll_rates = []
    mean_dists = []
    inv_rates = [0.5, 0.75, 1.0, 0.5, 0.75, 1.0]
    obses = [False, False, False, True, True, True]
    # PŘÍPRAVA MŘÍŽKY PRO 6 GRAFŮ (Těsně před for cyklem)
    fig_dists, axs_dists = plt.subplots(2, 3, figsize=(14, 8), sharey=True, dpi=300)
    axs_dists_flat = axs_dists.flatten() # Zploštíme mřížku pro snadné indexování (0 až 5)
    for i in range(6):
        inv_rate = inv_rates[i]
        obstacles = obses[i]
        seed_num = 420
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
        env = HerdingEnv(world_instance=world, sc=new_sc, test=True)
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
        obs, info = env.reset(inv_rate=inv_rate, obstacle=obstacles)
        print("Start")
        #running forever
        running = True
        whole_reward = 0 
        episode_num = 0
        render_every = 4
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
            # controlling visualizer window
            # if hasattr(vis, 'is_open') and not vis.is_open:
            #     print("Window closed, ending...")
            #     running = False
            #     break
            # rendering
            # if ep_len % render_every == 0:
            #     vis.render(state, world_instance=world)
            #restarting episode
            if terminated or truncated:
                #print(f"Episode over! Reward: {whole_reward:.1f}.")
                whole_reward = 0
                ep_len = 0
                episode_num += 1
                lock_all_seeds(seed_num + episode_num)
                if episode_num % 25 == 0:
                    print("Episode: " + str(episode_num))
                if episode_num == 100:
                    break
                #plt.pause(1.0)
                obs, info = env.reset(inv_rate=inv_rate, obstacle=obstacles)
                # if hasattr(vis, 'is_open') and not vis.is_open:
                #    vis.is_open = False
                # #visualizer
                # vis = MatplotlibVisualizer(sc_config=env.sc, _3d=True, quiver=False)
        purs_purs_crash = world.purs_purs_coll
        purs_obs_crash = world.purs_obs_coll
        purs_prime_crash = world.purs_prime_coll
        purs_gr_crash = world.purs_gr_coll
        inv_prime = world.inv_prime_coll
        lost = world.prime_crash
        dists = np.array(env.dist_in_the_end)
        dist_mean = np.mean(dists)
        print("Invader won: " + str(inv_prime))
        print("Pursuer crashed to Prime: " + str(purs_prime_crash))
        print("Pursuer crashed to Ground: " + str(purs_gr_crash))
        print("Pursuer crashed to Obstacle: " + str(purs_obs_crash))
        print("Pursuer crashed to Pursuer: " + str(purs_purs_crash))
        win_rate = 1 - (lost / float(episode_num))
        total_coll = purs_obs_crash + purs_prime_crash + purs_purs_crash + purs_gr_crash
        coll_rate = total_coll / float(episode_num)
        print("win rate: " + str(win_rate))
        print("total coll: " + str(total_coll) + " coll rate: " + str(coll_rate))
        print("Surface inv and prime dist in the end of won episodes: " + str(dist_mean))
        win_rates.append(win_rate)
        coll_rates.append(coll_rate)
        mean_dists.append(dist_mean)
        all_distances = env.all_dists
        if len(all_distances) > 0:
            dist_matrix = np.array(all_distances)
            mean_over_time = np.mean(dist_matrix, axis=0)
            std_over_time = np.std(dist_matrix, axis=0)
            steps = np.arange(dist_matrix.shape[1])
            
            # Vybereme konkrétní podgraf z naší 2x3 mřížky podle aktuálního kroku cyklu
            ax = axs_dists_flat[i]
            
            main_color = "#6D9EEB"
            ax.plot(steps, mean_over_time, label="Mean Distance", color=main_color, linewidth=1.5, zorder=3)
            ax.fill_between(steps, 
                             mean_over_time - std_over_time, 
                             mean_over_time + std_over_time, 
                             color=main_color, alpha=0.15, linewidth=0, label="Std Dev", zorder=2)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#DDDDDD')
            ax.spines['bottom'].set_color('#DDDDDD')
            ax.grid(True, linestyle="-", alpha=0.4, color='#EEEEEE', zorder=0)
            
            # Kratší titulky, aby se do mřížky hezky vešly
            obs_title = "With Obs" if obstacles else "No Obs"
            ax.set_title(f"Speed {inv_rate}x ({obs_title})", color='#333333', pad=10, fontsize=12)
            ax.set_xlabel("Steps", color='#555555', fontsize=10)
            
            # Osu Y popíšeme jen u prvních grafů v řádku, ať to není přeplácané
            if i % 3 == 0:
                ax.set_ylabel("Dist from Prime (m)", color='#555555', fontsize=10)
            
            # Legendu stačí ukázat jen v prvním grafu
            if i == 0:
                ax.legend(frameon=False, labelcolor='#555555', loc='upper right')
            # plt.show()
            #plt.show()
        if i == 5:
            plot_failure_histogram(env.step_fail, max_steps=250)
            
    fig_dists.tight_layout()
    fig_dists.savefig("herding_dists_combined_grid.png", bbox_inches='tight')
    plt.show() # nebo můžete plt.close(fig_dists), pokud nechcete aby to vyskakovalo
    # Helper function to generate bar charts to avoid code repetition
    def plot_comparison_bar_chart(data, title, ylabel, filename=None):
        labels = ['0.5x', '0.75x', '1.0x']
        
        data_no_obs = data[0:3]
        data_with_obs = data[3:6]
        
        x = np.arange(len(labels))
        width = 0.22    # Výrazně tenčí sloupce
        offset = 0.08   # Menší posun, aby se stále lehce překrývaly
        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        
        # Jemné, pastelové barvy
        color_no_obs = '#7EA6E0'  # Světlejší tlumená modrá
        color_with_obs = '#E07E7E' # Jemná lososová/červená
        
        # Vykreslení s bílým okrajem a zorderem (aby červená byla hezky vpředu)
        rects1 = ax.bar(x - offset, data_no_obs, width, label='Without Obstacles', 
                        color=color_no_obs, edgecolor='white', linewidth=1.0, zorder=3, alpha=0.9)
        rects2 = ax.bar(x + offset, data_with_obs, width, label='With Obstacles', 
                        color=color_with_obs, edgecolor='white', linewidth=1.0, zorder=4, alpha=0.9)
        
        # Očištění grafu od rámečků
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#DDDDDD')
        ax.spines['bottom'].set_color('#DDDDDD')
        
        # Jemná mřížka pod daty
        ax.grid(axis='y', linestyle='-', alpha=0.4, color='#EEEEEE', zorder=0)
        
        # Jemné šedé texty místo agresivní černé
        ax.set_ylabel(ylabel, color='#555555', labelpad=10)
        ax.set_xlabel('Target Speed (Multiplier of Pursuer Speed)', color='#555555', labelpad=10)
        ax.set_title(title, color='#333333', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, color='#555555')
        
        if "Rate" in ylabel and max(data) <= 1.0:
            ax.set_ylim([0, 1.15])
            
        # Legenda bez rámečku, nenápadná
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, labelcolor='#555555')
        
        # Decentní popisky
        ax.bar_label(rects1, padding=3, fmt='%.2f', fontsize=9, color='#666666')
        ax.bar_label(rects2, padding=3, fmt='%.2f', fontsize=9, color='#666666')
        
        fig.tight_layout()
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        # plt.show()

    # 1. Win Rate Chart
    plot_comparison_bar_chart(
        data=win_rates, 
        title='Herding Model: Win Rate Comparison', 
        ylabel='Win Rate', 
        filename='comparison_win_rate.png'
    )

    # 2. Collision Rate Chart
    plot_comparison_bar_chart(
        data=coll_rates, 
        title='Herding Model: Collision Rate Comparison', 
        ylabel='Collisions per Episode', 
        filename='comparison_coll_rate.png'
    )

    # 3. Mean Final Distance Chart
    plot_comparison_bar_chart(
        data=mean_dists, 
        title='Herding Model: Final Distance Comparison (Wins Only)', 
        ylabel='Distance from Base (m)', 
        filename='comparison_mean_dist.png'
    )

if __name__ == "__main__":
    test_herding_model()