import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Global style configuration
sns.set_theme(style="whitegrid")
mpl.rcParams['figure.dpi'] = 300

def plot_parametric_comparison(data_dict, behaviors, scenario_title, y_label, filename):
    """
    Plots a side-by-side bar chart comparing metrics with and without obstacles.
    """
    x = np.arange(len(behaviors))
    width = 0.20  # Zúžená šířka sloupců
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    
    # Custom color palette 
    colors = ['#4A90E2', '#50C878', '#FF9F43']
    models = list(data_dict.keys())

    for i, model in enumerate(models):
        # Data without obstacles (first 2 values: Kamikaze, Evader)
        no_obs = data_dict[model][:2]
        ax1.bar(x + (i - 1) * width, no_obs, width, label=model, 
                color=colors[i], edgecolor='white', alpha=0.9)
        
        # Data with obstacles (last 2 values: Kamikaze, Evader)
        with_obs = data_dict[model][2:]
        ax2.bar(x + (i - 1) * width, with_obs, width, label=model, 
                color=colors[i], edgecolor='white', alpha=0.9)

    # Titles with automatic scenario insertion
    ax1.set_title(f'Scenario {scenario_title}: Without Obstacles', fontsize=16, pad=15)
    ax2.set_title(f'Scenario {scenario_title}: With Obstacles', fontsize=16, pad=15)
    
    ax1.set_ylabel(y_label, fontsize=13)
    for ax in [ax1, ax2]:
        ax.set_xlabel('Invader Behavior', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(behaviors)
        
        # Dynamic y-axis height adjustment
        if y_label == 'Win Rate':
            ax.set_ylim(0, 1.1) 
        else:
            max_val = max([max(v) for v in data_dict.values()])
            ax.set_ylim(0, max_val * 1.2) # Leave space for text labels above bars
            
        sns.despine(ax=ax) # Remove top and right borders

        # Annotate values directly above the bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=9)

    # Global legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    behaviors = ['Kamikaze', 'Evader']
    # Format: [NoObs_Kamikaze, NoObs_Evader, WithObs_Kamikaze, WithObs_Evader]
    data_wr_5v1 = {
        'Baseline': [0.980, 0.990, 0.950, 0.985],
        'Variant 1': [0.685, 0.990, 0.685, 0.970],
        'Variant 2': [0.930, 0.995, 0.875, 0.960]
    }
    
    data_cr_5v1 = {
        'Baseline': [0.000, 0.000, 0.100, 0.105],
        'Variant 1': [0.110, 0.080, 0.385, 0.470],
        'Variant 2': [0.010, 0.010, 0.195, 0.195]
    }
    
    data_wr_20v8 = {
        'Baseline': [0.855, 0.890, 0.735, 0.850],
        'Variant 1': [0.005, 0.585, 0.005, 0.430],
        'Variant 2': [0.775, 0.820, 0.455, 0.635]
    }
    
    data_cr_20v8 = {
        'Baseline': [0.780, 0.925, 1.400, 1.875],
        'Variant 1': [1.270, 2.260, 3.115, 4.715],
        'Variant 2': [1.930, 2.200, 5.125, 5.615]
    }

    # Generate plots for 5 vs 1
    plot_parametric_comparison(data_wr_5v1, behaviors, "5 vs 1", "Win Rate", "win_rate_def_5v1.png")
    plot_parametric_comparison(data_cr_5v1, behaviors, "5 vs 1", "Collision Rate", "coll_rate_def_5v1.png")
    # Generate plots for 20 vs 8
    plot_parametric_comparison(data_wr_20v8, behaviors, "20 vs 8", "Win Rate", "win_rate_def_20v8.png")
    plot_parametric_comparison(data_cr_20v8, behaviors, "20 vs 8", "Collision Rate", "coll_rate_def_20v8.png")
    
    # You can copy the blocks above for 20 vs 8, just change the data and filenames
    print("Plots successfully generated and saved!")