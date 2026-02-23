import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from pursuer_states import States

class MatplotlibVisualizer:
    def __init__(self, sc_config, _3d=False, quiver=False):
        #basic configs
        self.sc = sc_config
        self._3d = _3d
        self.quiver = quiver
        
        self.p_dots = []
        self.p_paths = []
        self.i_dots = []
        self.i_paths = []
        self.u_dot = None
        self.u_path = None
        self.obs_patch = None
        # self.fig = self.sc.fig
        # self.ax = self.sc.ax
        if not _3d:
            self._init_plot2D()
        else:
            self._init_plot3D()
        #boolean to signal plt window close
        self.is_open = True
        #manual control
        self.manual_vel = np.zeros(3) if self._3d else np.zeros(2)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self._on_key_release)
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        
        plt.ion()
        plt.show(block=False)
        
    def _init_plot3D(self):
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        #space size
        self.ax.set_xlim(-1, self.sc.WORLD_WIDTH)
        self.ax.set_ylim(-1, self.sc.WORLD_HEIGHT)
        self.ax.set_zlim(-1, self.sc.WORLD_Z)

        #keeping ratio real
        self.ax.set_box_aspect((self.sc.WORLD_WIDTH, self.sc.WORLD_HEIGHT, self.sc.WORLD_Z))

        #view of camero
        self.ax.view_init(elev=10, azim=120)

        #grids, ticks
        self.ax.grid(True, which='major', color='#b0b0b0', linestyle='-', linewidth=0.7)
        self.ax.minorticks_on()
        self.ax.tick_params(axis='both', which='both', labelsize=8, colors='#555')
        #pursuers
        for _ in range(self.sc.PURSUER_NUM):
            p_dot, = self.ax.plot([], [], 'o', color='#d62728', label='Pursuer', markersize=self.sc.drone_marker_size)
            self.p_dots.append(p_dot)
            #p_path, = self.ax.plot([], [], '--', color='#d62728', alpha=0.6, linewidth=1.5)
            #self.p_paths.append(p_path)
        #invaders
        for _ in range(self.sc.INVADER_NUM):
            i_dot, = self.ax.plot([], [], 'o', color='#1f77b4', label='Invader', markersize=self.sc.drone_marker_size)
            self.i_dots.append(i_dot)
            #i_path, = self.ax.plot([], [], '--', color='#1f77b4', alpha=0.6, linewidth=1.5)
            #self.i_paths.append(i_path)
        #prime
        self.u_dot, = self.ax.plot([], [], 'o', color="#10ec22", label='Prime Unit', markersize=self.sc.prime_marker_size)
        #self.u_path, = self.ax.plot([], [], '--', color="#77cc70", alpha=0.6, linewidth=3.0)
        #obstacle
        if self.sc.obstacle:
            self.obs_patch = []
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            for i in range(len(self.sc.obs_rads)):
                obs_pos = self.sc.obs_pos[i]
                obs_rad = self.sc.obs_rads[i]
                x = obs_pos[0] + obs_rad * np.cos(u) * np.sin(v)
                y = obs_pos[1] + obs_rad * np.sin(u) * np.sin(v)
                z = obs_pos[2] + obs_rad * np.cos(v)
                self.obs_patch.append(self.ax.plot_wireframe(x, y, z, color="black"))
        
    def _init_plot2D(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        #space size
        self.ax.set_xlim(-1, self.sc.WORLD_WIDTH)
        self.ax.set_ylim(-1, self.sc.WORLD_HEIGHT)
        self.ax.set_aspect('equal')
        
        self.fig.canvas.draw() 

        #setting world grey, ticks and grid
        self.ax.set_facecolor("#f0f0f0")
        self.ax.set_xticks(np.arange(-1, self.sc.WORLD_WIDTH + 0.5, 2))
        self.ax.set_yticks(np.arange(-1, self.sc.WORLD_HEIGHT + 0.5, 2))
        self.ax.grid(True, which='major', color='#b0b0b0', linestyle='-', linewidth=0.7)
        self.ax.grid(True, which='minor', color='#d0d0d0', linestyle='-', linewidth=0.5)
        self.ax.minorticks_on()
        self.ax.tick_params(axis='both', which='both', labelsize=8, colors='#555')

        #making dots relative to size of the world
        trans = self.ax.transData.transform
        (x0, y0) = trans((0, 0))
        (x1, y1) = trans((self.sc.DRONE_RAD, 0))
        self.drone_in_pixels = np.hypot(x1 - x0, y1 - y0)
        self.prime_in_pixels = self.drone_in_pixels * (self.sc.UNIT_RAD/self.sc.DRONE_RAD)
        #self.obs_in_pixels = self.drone_in_pixels * (self.OBS_RAD/self.DRONE_RAD)
        #pursuers
        for _ in range(self.sc.PURSUER_NUM):
            p_dot, = self.ax.plot([], [], 'o', color='#d62728', label='Pursuer', markersize=self.drone_in_pixels)
            self.p_dots.append(p_dot)
            #p_path, = self.ax.plot([], [], '--', color='#d62728', alpha=0.6, linewidth=1.5)
            #self.p_paths.append(p_path)
        #invaders
        for _ in range(self.sc.INVADER_NUM):
            i_dot, = self.ax.plot([], [], 'o', color='#1f77b4', label='Invader', markersize=self.drone_in_pixels)
            self.i_dots.append(i_dot)
            #i_path, = self.ax.plot([], [], '--', color='#1f77b4', alpha=0.6, linewidth=1.5)
            #self.i_paths.append(i_path)
        #prime
        self.u_dot, = self.ax.plot([], [], 'o', color="#10ec22", label='Prime Unit', markersize=self.prime_in_pixels)
        #self.u_path, = self.ax.plot([], [], '--', color="#77cc70", alpha=0.6, linewidth=3.0)
        #obstacle
        if self.sc.obstacle:
            self.obs_patch = []
            for i in range(len(self.sc.obs_rads)):
                obs_pos = self.sc.obs_pos[i]
                obs_rad = self.sc.obs_rads[i]
                obs_patch = Circle(obs_pos, obs_rad, color='black', zorder=5)
                self.obs_patch.append(self.ax.add_patch(obs_patch))
        
        if self.quiver:
            #visual for prime vortex field
            x = np.linspace(-self.sc.FIELD_SIZE, self.sc.FIELD_SIZE, self.sc.FIELD_RES)
            y = np.linspace(-self.sc.FIELD_SIZE, self.sc.FIELD_SIZE, self.sc.FIELD_RES)
            self.grid_x_base, self.grid_y_base = np.meshgrid(x, y)
            self.quiver = self.ax.quiver(
                self.grid_x_base, self.grid_y_base, 
                np.zeros_like(self.grid_x_base), np.zeros_like(self.grid_y_base),
                color="#6fa840", alpha=0.7, pivot='mid', scale=20, width=0.003
            )
            #visual for invaders vortex field
            self.inv_quiver = []
            self.inv_grid_x_base, self.inv_grid_y_base = [], []
            x = np.linspace(-self.sc.INV_FIELD_SIZE, self.sc.INV_FIELD_SIZE, self.sc.INV_FIELD_RES)
            y = np.linspace(-self.sc.INV_FIELD_SIZE, self.sc.INV_FIELD_SIZE, self.sc.INV_FIELD_RES)
            for i in range(self.sc.INVADER_NUM):
                grid_x, grid_y = np.meshgrid(x, y)
                self.inv_grid_x_base.append(grid_x)
                self.inv_grid_y_base.append(grid_y)
                quiver = self.ax.quiver(
                self.inv_grid_x_base[i], self.inv_grid_y_base[i], 
                np.zeros_like(self.inv_grid_x_base[i]), np.zeros_like(self.inv_grid_y_base[i]),
                color="#244163", alpha=0.4, pivot='mid', scale=20, width=0.003
                )
                self.inv_quiver.append(quiver)

    def render(self, state, world_instance=None):
        #pursuers
        for i, (p_pos, p_state) in enumerate(zip(state['pursuers'], state['pursuers_status'])):
            p_dot = self.p_dots[i]
            p_dot.set_data([p_pos[0]], [p_pos[1]])
            if self._3d:
                p_dot.set_3d_properties([p_pos[2]])
            #color according to the state
            if p_state[0] == States.PURSUE:
                if p_state[1][1] == 1:
                    p_dot.set_color("#000000")
                else:
                    p_dot.set_color("#FFEE00")
            else:
                p_dot.set_color('#d62728')
                
        #invaders
        for i, i_pos in enumerate(state['invaders']):
            i_dot = self.i_dots[i]
            i_dot.set_data([i_pos[0]], [i_pos[1]])
            if self._3d:
                i_dot.set_3d_properties([i_pos[2]])
        #prime
        u_pos = state['prime']
        self.u_dot.set_data([u_pos[0]], [u_pos[1]])
        if self._3d:
            self.u_dot.set_3d_properties([u_pos[2]])
        #vortex field
        if not self._3d and world_instance is not None and self.quiver:
             self._update_vector_field(world_instance)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _update_vector_field(self, world):
        #making quiver graph, visualizing vortex field
        unit = world.prime
        center = unit.position
        grid_x = self.grid_x_base + center[0]
        grid_y = self.grid_y_base + center[1]
        u_arrows = []
        v_arrows = []
        if not world.pursuers:
            return
        ref_pursuer = world.pursuers[0]
        flat_x = grid_x.flatten()
        flat_y = grid_y.flatten()
        for x, y in zip(flat_x, flat_y):
            test_pos = np.array([x, y])
            force_vec = ref_pursuer.form_vortex_field_circle(unit, mock_position=test_pos, obstacle=ref_pursuer.curr_obs)
            norm = np.linalg.norm(force_vec)
            if norm > 0:
                force_vec = force_vec / norm
            u_arrows.append(force_vec[0])
            v_arrows.append(force_vec[1])
        new_offsets = np.column_stack((flat_x, flat_y))
        self.quiver.set_offsets(new_offsets)
        self.quiver.set_UVC(u_arrows, v_arrows)
        #quiver graph for invaders
        for idx, unit in enumerate(world.invaders):
            center = unit.position
            grid_x = self.inv_grid_x_base[idx] + center[0]
            grid_y = self.inv_grid_y_base[idx] + center[1]
            u_arrows = []
            v_arrows = []
            ref_pursuer = world.pursuers[0]
            flat_x = grid_x.flatten()
            flat_y = grid_y.flatten()
            for x, y in zip(flat_x, flat_y):
                test_pos = np.array([x, y])
                force_vec = ref_pursuer.pursuit_circling([unit, 0], mock_position=test_pos)
                norm = np.linalg.norm(force_vec)
                if norm > 0:
                    force_vec = force_vec / norm
                u_arrows.append(force_vec[0])
                v_arrows.append(force_vec[1])
            new_offsets = np.column_stack((flat_x, flat_y))
            self.inv_quiver[idx].set_offsets(new_offsets)
            self.inv_quiver[idx].set_UVC(u_arrows, v_arrows)

    def _on_close(self, event):
        #window of plt was closed
        self.is_open = False

    def _on_key_press(self, event):
        #keys pressed
        if event.key == 'up': self.manual_vel[1] = 1.0
        elif event.key == 'down': self.manual_vel[1] = -1.0
        elif event.key == 'left': self.manual_vel[0] = -1.0
        elif event.key == 'right': self.manual_vel[0] = 1.0
        if self._3d:
            if event.key == 'w': self.manual_vel[2] = 1.0
            elif event.key == 's': self.manual_vel[2] = -1.0

    def _on_key_release(self, event):
        #nulling the velocities on key release
        if event.key in ['up', 'down']: self.manual_vel[1] = 0.0
        elif event.key in ['left', 'right']: self.manual_vel[0] = 0.0
        if self._3d and event.key in ['w', 's']: self.manual_vel[2] = 0.0