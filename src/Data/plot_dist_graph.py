import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Global style configuration
sns.set_theme(style="whitegrid")
mpl.rcParams['figure.dpi'] = 300

def plot_triple_distance_comparison(data_dict, filename="comparison_dist_rate_4vs1_triplet.png"):
    """
    Plots three subplots showing the distance evolution over time for each model.
    """
    models = list(data_dict.keys())
    colors = ['#4A90E2', '#50C878', '#FF9F43'] 
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    # Main title configuration
    fig.suptitle('Invader-Prime Distance Comparison 4 vs 1 (Speed 1.0x, No Obstacles)', 
                 fontsize=18, fontweight='bold', y=1.02)

    for i, model in enumerate(models):
        ax = axs[i]
        # Data processing: compute mean and standard deviation
        data = np.array(data_dict[model])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        time = np.arange(len(mean)) * 0.1 # Convert steps to seconds (dt=0.1)
        
        # Plot shaded area for standard deviation (variance)
        ax.fill_between(time, mean - std, mean + std, color=colors[i], alpha=0.15)
        
        # Plot the main average distance line
        ax.plot(time, mean, color=colors[i], linewidth=2.5, label=f'Mean {model}')
        
        ax.set_title(model, fontsize=15, pad=10)
        ax.set_xlabel('Time [s]', fontsize=13)
        ax.set_ylim(0, 45)
        sns.despine(ax=ax) # Clean, modern look without a box frame
        
        if i == 0:
            ax.set_ylabel('Distance to Prime [m]', fontsize=13)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    #loading data
    data_A = np.load('RL_Herding/distances_Model_A_4_vs_1_rate_1.0_obs_False.npy')
    data_B = np.load('RL_Herding/distances_Model_B_4_vs_1_rate_1.0_obs_False.npy')
    data_C = np.load('RL_Herding/distances_Model_C_4_vs_1_rate_1.0_obs_False.npy')
    data_dict = {'Model A': data_A, 'Model B': data_B, 'Model C': data_C}
    plot_triple_distance_comparison(data_dict)