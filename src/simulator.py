import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from pursuer import Pursuer
from invader import Invader
from prime_unit import Prime_unit
from matplotlib.animation import FuncAnimation
#from scipy.optimize import linear_sum_assignment
from pursuer_states import States
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from prime_mode import Modes

class DroneSimulation:
    def __init__(self, sc_config, _3d=False, purs_acc=None, prime_acc=None, inv_acc=None, inv_control=False, 
                 prime_pos=None, inv_pos=None, purs_pos=None, crash_enabled=100, formation_delay=100, prime_mode=Modes.LINE):
        self._3d = _3d
        self.anim = None
        #figure and plots init
        self.sc = sc_config
        self.ax = sc_config.ax
        self.fig = sc_config.fig
        
        #init of agents
        self.prime = None
        self.prime_mode = prime_mode
        self.pursuers = []
        self.invaders = []
        self.captured_count = 0
        
        #agents path history
        self.hist_prime = []
        self.hist_pursuers = [[] for _ in range(self.sc.PURSUER_NUM)]
        self.hist_invaders = [[] for _ in range(self.sc.INVADER_NUM)]
        
        #init of agents
        self._init_agents(purs_acc, prime_acc, inv_acc, prime_pos, inv_pos, purs_pos)
        self.crash_enabled = crash_enabled
        self.purs_crash = False
        self.formation_delay = formation_delay
        self.start_anim = False
        
        #init of graphics
        #self._init_graphics()
        
        #manual control
        self.manual_control = inv_control
        if self.manual_control:
            self.manual_vel = np.zeros(3) if self._3d else np.zeros(2)
            #connecting key events
            self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
            self.fig.canvas.mpl_connect('key_release_event', self._on_key_release)

    def _init_agents(self, purs_acc, prime_acc, inv_acc, prime_pos, inv_pos, purs_pos):
        #borders and waypoint for prime unit
        x_border = self.sc.WORLD_WIDTH / 6
        y_border = self.sc.WORLD_HEIGHT / 6
        if self._3d:
            z_border = self.sc.WORLD_Z/6
            self.way_point = np.array([self.sc.WORLD_WIDTH - x_border, self.sc.WORLD_HEIGHT - y_border, self.sc.WORLD_Z - z_border])
        else:
            if self.prime_mode == Modes.CIRCLE:
                self.way_point = np.array([15.0, 15.0])
            elif self.prime_mode == Modes.LINE:
                self.way_point = np.array([self.sc.WORLD_WIDTH - x_border, self.sc.WORLD_HEIGHT - y_border])
        
        #prime unit init
        acc_prime = prime_acc or 0.1
        if self._3d:
            pos_u = prime_pos if prime_pos is not None else np.array([3.0, 3.0, 3.0])
        else:
            pos_u = prime_pos if prime_pos is not None else np.array([3.0, 3.0])
        self.prime = Prime_unit(position=pos_u, max_acc=acc_prime, max_omega=1.0)
        self.hist_prime.append(pos_u)

        #init positions and acceleration of agents (random)
        if self._3d:
            rnd_points_purs = purs_pos if purs_pos is not None else np.random.uniform(
                low=[-self.sc.PURSUER_NUM/2 - 2 + pos_u[0], -self.sc.PURSUER_NUM/2 - 2 + pos_u[1], -self.sc.PURSUER_NUM/2 - 2 + pos_u[2]], 
                high=[self.sc.PURSUER_NUM/2 + 2 + pos_u[0], self.sc.PURSUER_NUM/2 + 2 + pos_u[1], self.sc.PURSUER_NUM/2 + 2 +  pos_u[2]], 
                size=(self.sc.PURSUER_NUM, 3)
            )
            rnd_points_inv = inv_pos if inv_pos is not None else np.random.uniform(
                low=[self.sc.PURSUER_NUM/2 + 2 + pos_u[0], self.sc.PURSUER_NUM/2 + 2 + pos_u[1], pos_u[2]], 
                high=[self.sc.WORLD_WIDTH - x_border, self.sc.WORLD_HEIGHT - y_border, pos_u[2]], 
                size=(self.sc.INVADER_NUM, 3)
            )
        else:
            rnd_points_purs = purs_pos if purs_pos is not None else np.random.uniform(
                low=[-self.sc.PURSUER_NUM/2 - 2 + pos_u[0], -self.sc.PURSUER_NUM/2 - 2 + pos_u[1]], 
                high=[self.sc.PURSUER_NUM/2 + 2 + pos_u[0], self.sc.PURSUER_NUM/2 + 2 + pos_u[1]], 
                size=(self.sc.PURSUER_NUM, 2)
            )
            rnd_points_inv = inv_pos if inv_pos is not None else np.random.uniform(
                low=[self.sc.PURSUER_NUM/2 + 2 + pos_u[0], self.sc.PURSUER_NUM/2 + 2 + pos_u[1]], 
                high=[self.sc.WORLD_WIDTH - x_border, self.sc.WORLD_HEIGHT - y_border], 
                size=(self.sc.INVADER_NUM, 2)
            )
        rnd_acc_inv = np.full(self.sc.INVADER_NUM, inv_acc) if inv_acc is not None else np.random.uniform(low=1.3, high=1.5, size=(self.sc.INVADER_NUM,))
        acc_purs = purs_acc or 1.0
        #pursuer init
        for i in range(self.sc.PURSUER_NUM):
            p = Pursuer(position=rnd_points_purs[i], max_acc=acc_purs, max_omega=1.5, num=i, purs_num=self.sc.PURSUER_NUM)
            self.pursuers.append(p)
            self.hist_pursuers[i].append(rnd_points_purs[i])

        #invader init
        for i in range(self.sc.INVADER_NUM):
            inv = Invader(position=rnd_points_inv[i], max_acc=rnd_acc_inv[i], max_omega=1.5)
            self.invaders.append(inv)
            self.hist_invaders[i].append(rnd_points_inv[i])

    def _init_graphics(self):
        #ellipse for vortex field visualization
        if not self._3d:
            self.ellipse_patch = Ellipse(
                self.prime.center, 
                self.prime.axis_a*2, 
                self.prime.axis_b*2, 
                angle=self.prime.rot_angle*(180/np.pi),
                edgecolor="#10ec22", facecolor='none', lw=0.5, linestyle='--'
            )
            self.ax.add_patch(self.ellipse_patch)
        return

    def _update_ellipse(self):
        #updating the ellipse
        self.ellipse_patch.center = self.prime.center
        self.ellipse_patch.width = self.prime.axis_a * 2
        self.ellipse_patch.height = self.prime.axis_b * 2
        self.ellipse_patch.angle = self.prime.rot_angle * (180 / np.pi)
        
    def _update_vector_field_3D(self):
        #making quiver graph, visualizing vortex field in 3D
        unit = self.prime
        center = unit.position
        span = np.linspace(-self.sc.FIELD_SIZE, self.sc.FIELD_SIZE, self.sc.FIELD_RES)
        dx, dy = np.meshgrid(span, span)
        grid_x = center[0] + dx
        grid_y = center[1] + dy
        grid_z = np.full_like(grid_x, center[2])
        u_arrows = []
        v_arrows = []
        w_arrows = []
        if not self.pursuers:
            return
        ref_pursuer = self.pursuers[0] 
        rows, cols = grid_x.shape
        for r in range(rows):
            for c in range(cols):
                test_pos = np.array([grid_x[r, c], grid_y[r, c], center[2]])
                force_vec = ref_pursuer.form_vortex_field_circle(unit, mock_position=test_pos)
                norm = np.linalg.norm(force_vec)
                if norm > 0:
                    force_vec = force_vec / norm * 0.8
                u_arrows.append(force_vec[0])
                v_arrows.append(force_vec[1])
                w_arrows.append(0)
        if self.sc.quiver is not None:
            self.sc.quiver.remove()
        self.sc.quiver = self.ax.quiver(
            grid_x.flatten(), grid_y.flatten(), grid_z.flatten(),
            u_arrows, v_arrows, w_arrows,
            normalize=False, color='#6fa840', alpha=0.7, 
            arrow_length_ratio=0.3, linewidths=1.5, zorder=10
        )
        
    def _update_vector_field(self):
        #making quiver graph, visualizing vortex field
        unit = self.prime
        center = unit.position
        grid_x = self.sc.grid_x_base + center[0]
        grid_y = self.sc.grid_y_base + center[1]
        u_arrows = []
        v_arrows = []
        if not self.pursuers:
            return
        ref_pursuer = self.pursuers[0]
        flat_x = grid_x.flatten()
        flat_y = grid_y.flatten()
        for x, y in zip(flat_x, flat_y):
            test_pos = np.array([x, y])
            force_vec = ref_pursuer.form_vortex_field_circle(unit, mock_position=test_pos)
            norm = np.linalg.norm(force_vec)
            if norm > 0:
                force_vec = force_vec / norm
            u_arrows.append(force_vec[0])
            v_arrows.append(force_vec[1])
        new_offsets = np.column_stack((flat_x, flat_y))
        self.sc.quiver.set_offsets(new_offsets)
        self.sc.quiver.set_UVC(u_arrows, v_arrows)
            
    def _on_key_press(self, event):
        # controls on arrows
        if event.key == 'up':
            self.manual_vel[1] = 1.0
        elif event.key == 'down':
            self.manual_vel[1] = -1.0
        elif event.key == 'left':
            self.manual_vel[0] = -1.0
        elif event.key == 'right':
            self.manual_vel[0] = 1.0
        
        #for 3D up and down with w and s
        if self._3d:
            if event.key == 'w':
                self.manual_vel[2] = 1.0
            elif event.key == 's':
                self.manual_vel[2] = -1.0

    def _on_key_release(self, event):
        #on key release, velocity is zero
        if event.key in ['up', 'down']:
            self.manual_vel[1] = 0.0
        elif event.key in ['left', 'right']:
            self.manual_vel[0] = 0.0
        
        if self._3d and event.key in ['w', 's']:
            self.manual_vel[2] = 0.0

    def update(self, frame):
        #lists of not crashed drones
        free_inv = [inv for inv in self.invaders if not inv.crashed]
        free_purs = [pur for pur in self.pursuers if not pur.crashed]

        #AI path decision + manual
        dirs_i = []
        for idx, inv in enumerate(self.invaders):
            #if first invader is not captured and is manually controlled
            if idx == 0 and not inv.crashed and self.manual_control:
                #getting vector from keyboard
                input_acc = self.manual_vel.copy()
                acc_size = np.linalg.norm(input_acc)
                if acc_size > 0:
                    #giving max acc
                    inv_acc = (input_acc / acc_size) * inv.max_acc 
                else:
                    #no keys are pressed, zero acc
                    inv_acc = np.zeros_like(input_acc)
                dirs_i.append(inv_acc)
            #normal AI
            else:
                dirs_i.append(inv.evade(free_purs, self.prime))
        
        dirs_p = []
        for purs in self.pursuers:
            close_purs = [p for p in free_purs if np.linalg.norm(p.position - purs.position) <= self.sc.PURS_VIS]
            dirs_p.append(purs.pursue(free_inv, close_purs, self.prime))

        dir_u = self.prime.fly(self.way_point, free_inv, free_purs, self.prime_mode)

        #moving the drones
        for p, p_dir in zip(self.pursuers, dirs_p):
            p.move(p_dir)
        if frame > self.formation_delay or self.start_anim:
            self.start_anim = True
            for i, i_dir in zip(self.invaders, dirs_i):
                i.move(i_dir)
            self.prime.move(dir_u)

        #appending current position to their history
        for i, p in enumerate(self.pursuers):
            self.hist_pursuers[i].append(p.position.copy())
        for i, inv in enumerate(self.invaders):
            self.hist_invaders[i].append(inv.position.copy())
        self.hist_prime.append(self.prime.position.copy())

        #collisions check
        for p in free_purs:
            #capture check
            for i in free_inv:
                if np.sum((p.position - i.position)**2) < self.sc.CAPTURE_RAD**2 and not i.crashed:
                    self.captured_count += 1
                    i.crashed = True
                    if i.pursuer is not None:
                        i.pursuer.target = None
            #crash (pursuer and pursuer)
            if frame > self.crash_enabled or self.purs_crash:
                self.purs_crash = True
                for other in free_purs:
                    if (other is not p) and np.sum((p.position - other.position)**2) < self.sc.CRASH_RAD**2:
                        p.crashed = True
                        other.crashed = True
            #crash (pursuer and prime)
            if np.sum((p.position - self.prime.position)**2) < self.sc.UNIT_DOWN_RAD**2:
                self.prime.crashed = True
        #crash (invader and prime)
        for i in self.invaders:
            if not i.crashed and np.sum((i.position - self.prime.position)**2) < self.sc.UNIT_DOWN_RAD**2:
                self.prime.crashed = True

        #updating graphics
        #pursuers dots, paths
        for p_dot, p, hist in zip(self.sc.p_dots, self.pursuers, self.hist_pursuers):
            p_dot.set_data([p.position[0]], [p.position[1]])
            if self._3d:
                p_dot.set_3d_properties([p.position[2]])
            p_dot.set_color("#631616" if p.state == States.PURSUE else '#d62728')
            
            pos_arr = np.array(hist)
            idx = self.pursuers.index(p) 
            self.sc.p_paths[idx].set_data(pos_arr[:, 0], pos_arr[:, 1])
            if self._3d:
                self.sc.p_paths[idx].set_3d_properties(pos_arr[:,2])
        #invaders dots, paths
        for i_dot, i, hist in zip(self.sc.i_dots, self.invaders, self.hist_invaders):
            i_dot.set_data([i.position[0]], [i.position[1]])
            if self._3d:
                i_dot.set_3d_properties([i.position[2]])  
                  
            pos_arr = np.array(hist)
            idx = self.invaders.index(i)
            self.sc.i_paths[idx].set_data(pos_arr[:, 0], pos_arr[:, 1])
            if self._3d:
                self.sc.i_paths[idx].set_3d_properties(pos_arr[:,2])
        #prime dots, paths
        self.sc.u_dot.set_data([self.prime.position[0]], [self.prime.position[1]])
        if self._3d:
            self.sc.u_dot.set_3d_properties([self.prime.position[2]])  
        pos_arr_u = np.array(self.hist_prime)
        self.sc.u_path.set_data(pos_arr_u[:, 0], pos_arr_u[:, 1])
        if self._3d:
            self.sc.u_path.set_3d_properties(pos_arr_u[:,2])
        #vortex field
        if not self._3d:
            self._update_vector_field()
        #if frame % 5 == 0 and self._3d:
        #    self._update_vector_field_3D()
        #if all invaders are captured, or prime unit was taken down or has finished, the animation will end
        # if (self.captured_count >= self.sc.INVADER_NUM and self.prime.finished) or self.prime.crashed:
        #     if self.anim is not None:
        #         self.anim.event_source.stop()
        
        if frame == self.crash_enabled and not self.purs_crash:
            print("crash enabled!")
        if frame == self.formation_delay and not self.start_anim:
            print("anim_starts!")
        if self._3d:
            return self.sc.p_dots + self.sc.i_dots + self.sc.p_paths + self.sc.i_paths + \
                [self.sc.u_dot, self.sc.u_path], #self.sc.quiver]
        else:
            return self.sc.p_dots + self.sc.i_dots + self.sc.p_paths + self.sc.i_paths + \
                [self.sc.u_dot, self.sc.u_path, self.sc.quiver]

    def run(self):
        #running the animation
        if self._3d:
            self.anim = FuncAnimation(self.fig, self.update, frames=200, interval=50, blit=False)
        else:
            self.anim = FuncAnimation(self.fig, self.update, frames=200, interval=50, blit=True)
        plt.show()


# =============== CURRENTLY UNUSED CODE ======================
# def minimize_max_fast(cost):
#     vals = np.unique(cost)
#     lo, hi = 0, len(vals) - 1
#     best_T = None
#     best_assign = None
#     while lo <= hi:
#         mid = (lo + hi) // 2
#         T = vals[mid]
#         masked = np.where(cost <= T, cost, 1e9)
#         row_ind, col_ind = linear_sum_assignment(masked)
#         if (cost[row_ind, col_ind] <= T).all():
#             best_T = T
#             best_assign = list(col_ind)
#             hi = mid - 1
#         else:
#             lo = mid + 1
#     return best_assign

# def formation_calculator(purs: list[Pursuer], unit: Prime_unit, form_max: float):
#     #pursuers in state FORM and close to prime unit
#     form_ps = [p for p in purs if (p.state == States.FORM and np.linalg.norm(p.position - state["prime"].position) < form_max)]
#     n = len(form_ps)
#     if n == 0:
#         return
#     #radius and angles of the formation
#     formation_r = max(n*form_ps[0].dist_formation/(np.pi*2), form_ps[0].min_formation_r)
#     angle_piece = 2*np.pi/n
#     angles = np.arange(n)*angle_piece
#     #calculation of distance from every pos in formation to every drone
#     cx, cy = unit.position
#     form_pos = np.stack([cx + formation_r * np.cos(angles), cy + formation_r * np.sin(angles)], axis=1)
#     pos_now = np.array([pur.position for pur in form_ps])
#     diff = pos_now[:, np.newaxis, :] - form_pos[np.newaxis, :, :]
#     D = np.linalg.norm(diff, axis=2)
#     #minimax solver - that is finding the minimal maximal distance drone has to fly
#     col_idx = minimize_max_fast(D)
#     for i in range(n):
#         form_ps[i].num = col_idx[i]
#     return
