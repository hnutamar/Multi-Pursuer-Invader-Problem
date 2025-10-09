import matplotlib.pyplot as plt
import numpy as np
#Constants
WORLD_WIDTH = 20
WORLD_HEIGHT = 20
CAPTURE_RAD = 0.2
PURSUER_NUM = 3
INVADER_NUM = 1

fig, ax = plt.subplots(figsize=(8, 8))
#space size
ax.set_xlim(-1, WORLD_WIDTH)
ax.set_ylim(-1, WORLD_HEIGHT)
ax.set_aspect('equal')
#gray background
ax.set_facecolor("#f0f0f0")
#grid
ax.set_xticks(np.arange(-1, WORLD_WIDTH + 0.5, 2))
ax.set_yticks(np.arange(-1, WORLD_HEIGHT + 0.5, 2))
ax.grid(True, which='major', color='#b0b0b0', linestyle='-', linewidth=0.7)
ax.grid(True, which='minor', color='#d0d0d0', linestyle='-', linewidth=0.5)
ax.minorticks_on()
#params
ax.tick_params(axis='both', which='both', labelsize=8, colors='#555')
#drone and trajectory
p_dots = []
p_paths = []
for _ in range(PURSUER_NUM):
    p_dot, = ax.plot([], [], 'o', color='#d62728', label='Pursuer', markersize=6)
    p_dots.append(p_dot)
    p_path, = ax.plot([], [], '--', color='#d62728', alpha=0.6, linewidth=1.5)
    p_paths.append(p_path)

i_dot, = ax.plot([], [], 'o', color='#1f77b4', label='Invader', markersize=6)
i_path, = ax.plot([], [], '--', color='#1f77b4', alpha=0.6, linewidth=1.5)
#legend
leg = ax.legend(frameon=True)
leg.get_frame().set_edgecolor('#aaa')
leg.get_frame().set_facecolor('#fafafa')