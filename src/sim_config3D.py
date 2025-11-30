import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # Import pro 3D projekci

# Constants
WORLD_WIDTH = 30
WORLD_HEIGHT = 30
WORLD_Z = 10
DRONE_RAD = 0.3
UNIT_RAD = 0.6
PURSUER_NUM = 10
INVADER_NUM = 5
CAPTURE_RAD = 0.3
CRASH_RAD = 0.3
UNIT_DOWN_RAD = 0.6
#3D sim
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
#space size
ax.set_xlim(-1, WORLD_WIDTH)
ax.set_ylim(-1, WORLD_HEIGHT)
ax.set_zlim(-1, WORLD_Z)
ax.set_box_aspect((WORLD_WIDTH, WORLD_HEIGHT, WORLD_Z))
# Popisky os
# ax.set_xlabel('X Osa')
# ax.set_ylabel('Y Osa')
# ax.set_zlabel('Z Osa')
drone_marker_size = 3
unit_marker_size = 6
#view
ax.view_init(elev=10, azim=120) 
#grid
ax.grid(True, which='major', color='#b0b0b0', linestyle='-', linewidth=0.7)
ax.minorticks_on()
ax.tick_params(axis='both', which='both', labelsize=8, colors='#555')
#pursuer init
p_dots = []
p_paths = []
for _ in range(PURSUER_NUM):
    p_dot, = ax.plot([], [], [], 'o', color='#d62728', label='Pursuer', markersize=drone_marker_size)
    p_dots.append(p_dot)
    p_path, = ax.plot([], [], [], '--', color='#d62728', alpha=0.6, linewidth=1.5)
    p_paths.append(p_path)
#invader init
i_dots = []
i_paths = []
for _ in range(INVADER_NUM):
    i_dot, = ax.plot([], [], [], 'o', color='#1f77b4', label='Invader', markersize=drone_marker_size)
    i_dots.append(i_dot)
    i_path, = ax.plot([], [], [], '--', color='#1f77b4', alpha=0.6, linewidth=1.5)
    i_paths.append(i_path)
#prime unit init
u_dot, = ax.plot([], [], [], 'o', color="#10ec22", label='Prime Unit', markersize=unit_marker_size)
u_path, = ax.plot([], [], [], '--', color="#77cc70", alpha=0.6, linewidth=3.0)

#ax.legend(loc='lower left')
