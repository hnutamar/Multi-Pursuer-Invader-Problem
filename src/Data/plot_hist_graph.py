import seaborn as sns
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
# Čistý, minimalistický základ
plt.style.use('default')
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['figure.dpi'] = 300

def plot_pro_failure_histogram(failure_dict, filename="comparison_fail_4v1.png"):
    # Nastavení stylu - 'whitegrid' vypadá v bakalářce velmi čistě
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    colors = ['#4A90E2', '#50C878', '#FF9F43']
    
    # Projdeme modely a vykreslíme je
    for i, (model_name, times) in enumerate(failure_dict.items()):
        # kde=True přidá tu krásnou vyhlazenou čáru
        # element="step" nebo "bars" - bars je klasika
        sns.histplot(times, binwidth=1, kde=True, label=model_name, 
                     color=colors[i], alpha=0.3, edgecolor=colors[i], linewidth=1.5)

    plt.title('Failure Time Comparison 4 vs 1 (1.0x Speed, Obstacles)', fontsize=16, pad=20)
    plt.xlabel('Time to Failure [s]', fontsize=13)
    plt.ylabel('Frequency', fontsize=13)
    plt.xlim(0, 30)
    
    plt.legend(frameon=False, fontsize=12)
    sns.despine() # Odstraní horní a pravý okraj grafu pro moderní vzhled

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Načtení dat (předpokládejme, že jsi vybral scénář rate 1.0, bez překážek)
data_A = np.load('fails_Model_A_4_vs_1_rate_1.0_obs_True.npy')
data_B = np.load('fails_Model_B_4_vs_1_rate_1.0_obs_True.npy')
data_C = np.load('fails_Model_C_4_vs_1_rate_1.0_obs_True.npy')
data_dict = {'Model A': data_A, 'Model B': data_B, 'Model C': data_C}
dt = 0.1  # tvůj simulační krok

failure_dict_seconds = {}
for model, steps in data_dict.items():
    # Přepočet každého kroku na sekundy
    failure_dict_seconds[model] = [step * dt for step in steps]

# Teď už můžeš volat vykreslovací funkci s failure_dict_seconds
plot_pro_failure_histogram(failure_dict_seconds)