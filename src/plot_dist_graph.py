import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def plot_triple_distance_comparison(data_dict, filename="comparison_dist_rate_4vs1_triplet.png"):
    models = list(data_dict.keys())
    # Barvy konzistentní s předchozími grafy
    colors = ['#4A90E2', '#50C878', '#FF9F43'] 
    # Vytvoření 3 podgrafů vedle sebe (1 řádek, 3 sloupce)
    # sharey=True je klíčové pro férové srovnání!
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    fig.suptitle('Comparison of Invader Prime Distance 4 vs 1 (Speed 1.0x, No Obstacles)', 
                 fontsize=18, fontweight='bold', y=1.05)
    for i, model in enumerate(models):
        ax = axs[i]
        # Předpokládáme, že data jsou ve formátu (epizody, čas)
        data = np.array(data_dict[model])
        
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        time = np.arange(len(mean)) * 0.1 # Přepočet na sekundy (dt=0.02)
        
        # Vykreslení stínu (odchylky) - alpha=0.2 pro jemnost
        ax.fill_between(time, mean - std, mean + std, color=colors[i], alpha=0.2)
        
        # Vykreslení hlavní čáry (průměru)
        ax.plot(time, mean, color=colors[i], linewidth=2.5, label=f'Mean {model}')
        
        # Styling jednotlivých grafů
        ax.set_title(model, fontsize=14, fontweight='bold', color='#333333')
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Nastavení limitů - uprav podle tvých dat (např. 0 až 40m)
        ax.set_ylim(0, 45) 
        
        if i == 0:
            ax.set_ylabel('Distance to Prime [m]', fontsize=12)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

def graph():
    # Čistý, minimalistický základ
    plt.style.use('default')
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.titlesize'] = 13
    mpl.rcParams['figure.dpi'] = 300

    # Načtení dat (předpokládejme, že jsi vybral scénář rate 1.0, bez překážek)
    dist_A = np.load('distances_Model_A_4_vs_1_rate_1.0_obs_False.npy')
    dist_B = np.load('distances_Model_B_4_vs_1_rate_1.0_obs_False.npy')
    dist_C = np.load('distances_Model_C_4_vs_1_rate_1.0_obs_False.npy')

    # Výpočet průměru (přes všechny epizody)
    # Předpokládáme, že dist_A má tvar (epizody, časové_kroky)
    mean_A = np.mean(dist_A, axis=0)
    mean_B = np.mean(dist_B, axis=0)
    mean_C = np.mean(dist_C, axis=0)

    # Vykreslení
    plt.figure(figsize=(10, 6))
    plt.plot(mean_A, label='Model A', color='#4A90E2', linewidth=2)
    plt.plot(mean_B, label='Model B', color='#50C878', linewidth=2)
    plt.plot(mean_C, label='Model C', color='#FF9F43', linewidth=2)

    plt.xlabel('Time Step (0.02s)')
    plt.ylabel('Distance to Prime [m]')
    plt.title('Comparison of Invader Prime Distance 4 vs 1 (Speed 1.0x, No Obstacles)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("comparison_dist_rate_4vs1.png", dpi=300, bbox_inches='tight')
    plt.close()
    

# Načtení dat (předpokládejme, že jsi vybral scénář rate 1.0, bez překážek)
data_A = np.load('distances_Model_A_4_vs_1_rate_1.0_obs_False.npy')
data_B = np.load('distances_Model_B_4_vs_1_rate_1.0_obs_False.npy')
data_C = np.load('distances_Model_C_4_vs_1_rate_1.0_obs_False.npy')
data_dict = {'Model A': data_A, 'Model B': data_B, 'Model C': data_C}
plot_triple_distance_comparison(data_dict)