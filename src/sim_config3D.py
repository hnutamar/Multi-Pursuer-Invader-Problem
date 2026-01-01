import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class Sim3DConfig:
    def __init__(self, purs_num=15, inv_num=5, world_width=30, world_height=30, world_z=10, obstacle=False, obstacle_pos=np.array([10.0, 10.0, 5.0]), obstacle_rad=0.6):
        self.WORLD_WIDTH = world_width
        self.WORLD_HEIGHT = world_height
        self.WORLD_Z = world_z
        self.obstacle = obstacle
        self.OBS_POS = obstacle_pos
        self.OBS_RAD = obstacle_rad
        
        self.DRONE_RAD = 0.3
        self.UNIT_RAD = 0.6
        self.CAPTURE_RAD = 0.3
        self.CRASH_RAD = self.DRONE_RAD
        self.UNIT_DOWN_RAD = self.UNIT_RAD
        self.PURS_VIS = 4.0
        
        self.PURSUER_NUM = purs_num
        self.INVADER_NUM = inv_num

        self.drone_marker_size = self.DRONE_RAD * 10
        self.prime_marker_size = self.UNIT_RAD * 10

        self.p_dots = []
        self.p_paths = []
        self.i_dots = []
        self.i_paths = []
        self.u_dot = None
        self.u_path = None
        self.obs_patch = None
        
        self.FIELD_RES = 15
        self.FIELD_SIZE = 4

        self._init_plot()
        
        self.quiver = None
        
    def _init_plot(self):
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        #space size
        self.ax.set_xlim(-1, self.WORLD_WIDTH)
        self.ax.set_ylim(-1, self.WORLD_HEIGHT)
        self.ax.set_zlim(-1, self.WORLD_Z)

        #keeping ratio real
        self.ax.set_box_aspect((self.WORLD_WIDTH, self.WORLD_HEIGHT, self.WORLD_Z))

        #view of camero
        self.ax.view_init(elev=10, azim=120)

        #grids, ticks
        self.ax.grid(True, which='major', color='#b0b0b0', linestyle='-', linewidth=0.7)
        self.ax.minorticks_on()
        self.ax.tick_params(axis='both', which='both', labelsize=8, colors='#555')
        #pursuers
        for _ in range(self.PURSUER_NUM):
            p_dot, = self.ax.plot([], [], 'o', color='#d62728', label='Pursuer', markersize=self.drone_marker_size)
            self.p_dots.append(p_dot)
            p_path, = self.ax.plot([], [], '--', color='#d62728', alpha=0.6, linewidth=1.5)
            self.p_paths.append(p_path)
        #invaders
        for _ in range(self.INVADER_NUM):
            i_dot, = self.ax.plot([], [], 'o', color='#1f77b4', label='Invader', markersize=self.drone_marker_size)
            self.i_dots.append(i_dot)
            i_path, = self.ax.plot([], [], '--', color='#1f77b4', alpha=0.6, linewidth=1.5)
            self.i_paths.append(i_path)
        #prime
        self.u_dot, = self.ax.plot([], [], 'o', color="#10ec22", label='Prime Unit', markersize=self.prime_marker_size)
        self.u_path, = self.ax.plot([], [], '--', color="#77cc70", alpha=0.6, linewidth=3.0)
        #obstacle
        if self.obstacle:
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = self.OBS_POS[0] + self.OBS_RAD * np.cos(u) * np.sin(v)
            y = self.OBS_POS[1] + self.OBS_RAD * np.sin(u) * np.sin(v)
            z = self.OBS_POS[2] + self.OBS_RAD * np.cos(v)
            self.obs_patch = self.ax.plot_wireframe(x, y, z, color="black")
