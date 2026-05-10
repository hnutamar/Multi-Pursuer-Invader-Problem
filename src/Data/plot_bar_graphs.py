
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('default')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['figure.dpi'] = 300

def plot_master_comparison(data_dict, speeds, filename='comparison_coll_rate_4vs1.png'):
    x = np.arange(len(speeds))
    width = 0.25  #width of column
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    #colors of columns
    colors = ['#4A90E2', '#50C878', '#FF9F43']
    models = list(data_dict.keys())
    #left graph
    for i, model in enumerate(models):
        no_obs = data_dict[model][:3]
        ax1.bar(x + (i - 1) * width, no_obs, width, label=model, 
                color=colors[i], edgecolor='white', alpha=0.9, zorder=3)
    ax1.set_title('Scenario 4 vs 1: Without Obstacles', fontsize=14, pad=15)
    ax1.set_ylabel('Collision Rate', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(speeds)
    ax1.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    #right graph
    for i, model in enumerate(models):
        with_obs = data_dict[model][3:]
        ax2.bar(x + (i - 1) * width, with_obs, width, label=model, 
                color=colors[i], edgecolor='white', alpha=0.9, zorder=3)

    ax2.set_title('Scenario 4 vs 1: With Obstacles', fontsize=14, pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(speeds)
    ax2.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    #both
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Target Speed Multiplier', fontsize=12)
        ax.set_ylim(0, 0.8)
        #adding to top
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontsize=8)
    #legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    #data
    speeds = ['0.5x', '0.75x', '1.0x']
    #win rates
    data_wr1v1 = {
        'Model A': [0.99, 0.82, 0.70, 0.79, 0.55, 0.40],
        'Model B': [0.85, 0.60, 0.42, 0.79, 0.52, 0.34],
        'Model C': [1.00, 0.99, 0.85, 0.61, 0.56, 0.45]
    }

    data_wr4v1 = {
        'Model A': [0.94, 0.76, 0.72, 0.77, 0.65, 0.51],
        'Model B': [0.91, 0.74, 0.61, 0.79, 0.64, 0.51],
        'Model C': [0.96, 0.90, 0.69, 0.72, 0.59, 0.43]
    }
    #collision rates
    data_cr1v1 = {
        'Model A': [0.0, 0.01, 0.01, 0.08, 0.13, 0.12],
        'Model B': [0.0, 0.01, 0.01, 0.07, 0.10, 0.07],
        'Model C': [0.0, 0.0, 0.12, 0.12, 0.10, 0.15]
    }

    data_cr4v1 = {
        'Model A': [0.08, 0.12, 0.12, 0.64, 0.60, 0.52],
        'Model B': [0.10, 0.09, 0.1, 0.76, 0.65, 0.53],
        'Model C': [0.06, 0.09, 0.26, 0.61, 0.66, 0.67]
    }
    plot_master_comparison(data_cr4v1, speeds)