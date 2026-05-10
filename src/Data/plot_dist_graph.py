import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def plot_triple_distance_comparison(data_dict, filename="comparison_dist_rate_4vs1_triplet.png"):
    models = list(data_dict.keys())
    #colors of lines
    colors = ['#4A90E2', '#50C878', '#FF9F43'] 
    #three graphs
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle('Comparison of Invader Prime Distance 4 vs 1 (Speed 1.0x, No Obstacles)', 
                 fontsize=18, fontweight='bold', y=1.05)
    for i, model in enumerate(models):
        ax = axs[i]
        #episode - time
        data = np.array(data_dict[model])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        time = np.arange(len(mean)) * 0.1 #to secs
        #dev
        ax.fill_between(time, mean - std, mean + std, color=colors[i], alpha=0.2)
        #main line
        ax.plot(time, mean, color=colors[i], linewidth=2.5, label=f'Mean {model}')
        #styling graphs
        ax.set_title(model, fontsize=14, fontweight='bold', color='#333333')
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #limits
        ax.set_ylim(0, 45) 
        if i == 0:
            ax.set_ylabel('Distance to Prime [m]', fontsize=12)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    #loading data
    data_A = np.load('distances_Model_A_4_vs_1_rate_1.0_obs_False.npy')
    data_B = np.load('distances_Model_B_4_vs_1_rate_1.0_obs_False.npy')
    data_C = np.load('distances_Model_C_4_vs_1_rate_1.0_obs_False.npy')
    data_dict = {'Model A': data_A, 'Model B': data_B, 'Model C': data_C}
    plot_triple_distance_comparison(data_dict)