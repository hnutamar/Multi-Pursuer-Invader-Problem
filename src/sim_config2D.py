import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

class Sim2DConfig:
    def __init__(self, purs_num=15, inv_num=5, world_width=30, world_height=30, obstacle=False, obstacle_rad=0.6, obstacle_pos=[15.0, 15.0]):
        self.WORLD_WIDTH = world_width
        self.WORLD_HEIGHT = world_height
        self.obstacle = obstacle
        
        self.CAPTURE_RAD = 0.3
        self.DRONE_RAD = 0.3
        self.UNIT_RAD = 0.6
        self.obs_pos = obstacle_pos
        self.obs_rads = obstacle_rad
        
        self.PURSUER_NUM = purs_num
        self.INVADER_NUM = inv_num
        
        self.CRASH_RAD = self.DRONE_RAD
        self.UNIT_DOWN_RAD = self.UNIT_RAD
        #visibility of pursuer of other pursuers
        self.PURS_VIS = 2.0
        
        self.p_dots = []
        self.p_paths = []
        self.i_dots = []
        self.i_paths = []
        self.u_dot = None
        self.u_path = None
        self.obs_patch = None
        #vortex fields size
        self.FIELD_RES = 15
        self.FIELD_SIZE = 4
        self.INV_FIELD_RES = 8
        self.INV_FIELD_SIZE = 2

        self._init_plot()

    def _init_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        #space size
        self.ax.set_xlim(-1, self.WORLD_WIDTH)
        self.ax.set_ylim(-1, self.WORLD_HEIGHT)
        self.ax.set_aspect('equal')
        
        self.fig.canvas.draw() 

        #setting world grey, ticks and grid
        self.ax.set_facecolor("#f0f0f0")
        self.ax.set_xticks(np.arange(-1, self.WORLD_WIDTH + 0.5, 2))
        self.ax.set_yticks(np.arange(-1, self.WORLD_HEIGHT + 0.5, 2))
        self.ax.grid(True, which='major', color='#b0b0b0', linestyle='-', linewidth=0.7)
        self.ax.grid(True, which='minor', color='#d0d0d0', linestyle='-', linewidth=0.5)
        self.ax.minorticks_on()
        self.ax.tick_params(axis='both', which='both', labelsize=8, colors='#555')

        #making dots relative to size of the world
        trans = self.ax.transData.transform
        (x0, y0) = trans((0, 0))
        (x1, y1) = trans((self.DRONE_RAD, 0))
        self.drone_in_pixels = np.hypot(x1 - x0, y1 - y0)
        self.prime_in_pixels = self.drone_in_pixels * (self.UNIT_RAD/self.DRONE_RAD)
        #self.obs_in_pixels = self.drone_in_pixels * (self.OBS_RAD/self.DRONE_RAD)
        #pursuers
        for _ in range(self.PURSUER_NUM):
            p_dot, = self.ax.plot([], [], 'o', color='#d62728', label='Pursuer', markersize=self.drone_in_pixels)
            self.p_dots.append(p_dot)
            p_path, = self.ax.plot([], [], '--', color='#d62728', alpha=0.6, linewidth=1.5)
            self.p_paths.append(p_path)
        #invaders
        for _ in range(self.INVADER_NUM):
            i_dot, = self.ax.plot([], [], 'o', color='#1f77b4', label='Invader', markersize=self.drone_in_pixels)
            self.i_dots.append(i_dot)
            i_path, = self.ax.plot([], [], '--', color='#1f77b4', alpha=0.6, linewidth=1.5)
            self.i_paths.append(i_path)
        #prime
        self.u_dot, = self.ax.plot([], [], 'o', color="#10ec22", label='Prime Unit', markersize=self.prime_in_pixels)
        self.u_path, = self.ax.plot([], [], '--', color="#77cc70", alpha=0.6, linewidth=3.0)
        #obstacle
        if self.obstacle:
            self.obs_patch = []
            for i in range(len(self.obs_rads)):
                obs_pos = self.obs_pos[i]
                obs_rad = self.obs_rads[i]
                obs_patch = Circle(obs_pos, obs_rad, color='black', zorder=5)
                self.obs_patch.append(self.ax.add_patch(obs_patch))
        
        #visual for prime vortex field
        x = np.linspace(-self.FIELD_SIZE, self.FIELD_SIZE, self.FIELD_RES)
        y = np.linspace(-self.FIELD_SIZE, self.FIELD_SIZE, self.FIELD_RES)
        self.grid_x_base, self.grid_y_base = np.meshgrid(x, y)
        self.quiver = self.ax.quiver(
            self.grid_x_base, self.grid_y_base, 
            np.zeros_like(self.grid_x_base), np.zeros_like(self.grid_y_base),
            color="#6fa840", alpha=0.7, pivot='mid', scale=20, width=0.003
        )
        #visual for invaders vortex field
        self.inv_quiver = []
        self.inv_grid_x_base, self.inv_grid_y_base = [], []
        x = np.linspace(-self.INV_FIELD_SIZE, self.INV_FIELD_SIZE, self.INV_FIELD_RES)
        y = np.linspace(-self.INV_FIELD_SIZE, self.INV_FIELD_SIZE, self.INV_FIELD_RES)
        for i in range(self.INVADER_NUM):
            grid_x, grid_y = np.meshgrid(x, y)
            self.inv_grid_x_base.append(grid_x)
            self.inv_grid_y_base.append(grid_y)
            quiver = self.ax.quiver(
            self.inv_grid_x_base[i], self.inv_grid_y_base[i], 
            np.zeros_like(self.inv_grid_x_base[i]), np.zeros_like(self.inv_grid_y_base[i]),
            color="#244163", alpha=0.4, pivot='mid', scale=20, width=0.003
            )
            self.inv_quiver.append(quiver)