import matplotlib.pyplot as plt
import numpy as np
#Constants
WORLD_WIDTH = 30
WORLD_HEIGHT = 30
CAPTURE_RAD = 0.3
CRASH_RAD = 0.3
UNIT_DOWN_RAD = 0.6
PURSUER_NUM = 10
INVADER_NUM = 0

DRONE_RAD = 0.3
UNIT_RAD = 0.6

fig, ax = plt.subplots(figsize=(8, 8))
#space size
ax.set_xlim(-1, WORLD_WIDTH)
ax.set_ylim(-1, WORLD_HEIGHT)
ax.set_aspect('equal')
fig.canvas.draw()
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
trans = ax.transData.transform
(x0, y0) = trans((0, 0))
(x1, y1) = trans((DRONE_RAD, 0))
drone_in_pixels = np.hypot(x1 - x0, y1 - y0)
unit_in_pixels = drone_in_pixels * 2
p_dots = []
p_paths = []
for _ in range(PURSUER_NUM):
    p_dot, = ax.plot([], [], 'o', color='#d62728', label='Pursuer', markersize=drone_in_pixels)
    p_dots.append(p_dot)
    p_path, = ax.plot([], [], '--', color='#d62728', alpha=0.6, linewidth=1.5)
    p_paths.append(p_path)

i_dots = []
i_paths = []
for _ in range(INVADER_NUM):
    i_dot, = ax.plot([], [], 'o', color='#1f77b4', label='Invader', markersize=drone_in_pixels)
    i_dots.append(i_dot)
    i_path, = ax.plot([], [], '--', color='#1f77b4', alpha=0.6, linewidth=1.5)
    i_paths.append(i_path)

u_dot, = ax.plot([], [], 'o', color="#10ec22", label='Prime Unit', markersize=unit_in_pixels)
u_path, = ax.plot([], [], '--', color="#77cc70", alpha=0.6, linewidth=3.0)
#legend
#leg = ax.legend(frameon=True)
#leg.get_frame().set_edgecolor('#aaa')
#leg.get_frame().set_facecolor('#fafafa')