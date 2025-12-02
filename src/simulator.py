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

class DroneSimulation:
    def __init__(self, sc_config, _3d=False):
        self._3d = _3d
        #figure and plots init
        self.sc = sc_config
        self.ax = sc_config.ax
        self.fig = sc_config.fig
        
        #init of agents
        self.prime = None
        self.pursuers = []
        self.invaders = []
        self.captured_count = 0
        
        #agents path history
        self.hist_prime = []
        self.hist_pursuers = [[] for _ in range(self.sc.PURSUER_NUM)]
        self.hist_invaders = [[] for _ in range(self.sc.INVADER_NUM)]
        
        #init of agents
        self._init_agents()
        
        #init of graphics
        self._init_graphics()

    def _init_agents(self):
        #borders and waypoint for prime unit
        x_border = self.sc.WORLD_WIDTH / 6
        y_border = self.sc.WORLD_HEIGHT / 6
        if self._3d:
            z_border = self.sc.WORLD_Z/6
            self.way_point = np.array([self.sc.WORLD_WIDTH - x_border, self.sc.WORLD_HEIGHT - y_border, self.sc.WORLD_Z - z_border])
        else:
            self.way_point = np.array([self.sc.WORLD_WIDTH - x_border, self.sc.WORLD_HEIGHT - y_border])
        
        #prime unit init
        if self._3d:
            pos_u = [3.0, 3.0, 3.0]
        else:
            pos_u = [3.0, 3.0]
        self.prime = Prime_unit(position=pos_u, max_acc=0.1, max_omega=1.0)
        self.hist_prime.append(pos_u)

        #init positions and acceleration of agents (random)
        if self._3d:
            rnd_points_purs = np.random.uniform(
                low=[-self.sc.PURSUER_NUM/2 - 2 + pos_u[0], -self.sc.PURSUER_NUM/2 - 2 + pos_u[1], pos_u[2]], 
                high=[self.sc.PURSUER_NUM/2 + 2 + pos_u[0], self.sc.PURSUER_NUM/2 + 2 + pos_u[1], pos_u[2]], 
                size=(self.sc.PURSUER_NUM, 3)
            )
            rnd_points_inv = np.random.uniform(
                low=[self.sc.PURSUER_NUM/2 + 2 + pos_u[0], self.sc.PURSUER_NUM/2 + 2 + pos_u[1], pos_u[2]], 
                high=[self.sc.WORLD_WIDTH - x_border, self.sc.WORLD_HEIGHT - y_border, pos_u[2]], 
                size=(self.sc.INVADER_NUM, 3)
            )
        else:
            rnd_points_purs = np.random.uniform(
                low=[-self.sc.PURSUER_NUM/2 - 2 + pos_u[0], -self.sc.PURSUER_NUM/2 - 2 + pos_u[1]], 
                high=[self.sc.PURSUER_NUM/2 + 2 + pos_u[0], self.sc.PURSUER_NUM/2 + 2 + pos_u[1]], 
                size=(self.sc.PURSUER_NUM, 2)
            )
            rnd_points_inv = np.random.uniform(
                low=[self.sc.PURSUER_NUM/2 + 2 + pos_u[0], self.sc.PURSUER_NUM/2 + 2 + pos_u[1]], 
                high=[self.sc.WORLD_WIDTH - x_border, self.sc.WORLD_HEIGHT - y_border], 
                size=(self.sc.INVADER_NUM, 2)
            )
        rnd_acc_inv = np.random.uniform(low=0.9, high=1.7, size=(self.sc.INVADER_NUM,))

        #pursuer init
        for i in range(self.sc.PURSUER_NUM):
            p = Pursuer(position=rnd_points_purs[i], max_acc=0.8, max_omega=1.5, num=i, purs_num=self.sc.PURSUER_NUM)
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

    def update(self, frame):
        #lists of not crashed drones
        free_inv = [inv for inv in self.invaders if not inv.captured]
        free_purs = [pur for pur in self.pursuers if pur.state != States.CRASHED]

        #AI path decision
        dirs_i = [inv.evade(free_purs, self.prime) for inv in self.invaders]
        
        dirs_p = []
        for purs in self.pursuers:
            close_purs = [p for p in free_purs if np.linalg.norm(p.position - purs.position) <= purs.vis_r]
            dirs_p.append(purs.pursue(free_inv, close_purs, self.prime))

        dir_u = self.prime.fly(self.way_point, free_inv, free_purs)

        #moving the drones
        for p, p_dir in zip(self.pursuers, dirs_p):
            p.move(p_dir)
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
                if np.sum((p.position - i.position)**2) < self.sc.CAPTURE_RAD**2 and not i.captured:
                    self.captured_count += 1
                    i.captured = True
                    if i.pursuer is not None:
                        i.pursuer.target = None
            #crash (pursuer and pursuer)
            for other in free_purs:
                if (other is not p) and np.sum((p.position - other.position)**2) < self.sc.CRASH_RAD**2:
                    p.state = States.CRASHED
                    other.state = States.CRASHED
            #crash (pursuer and prime)
            if np.sum((p.position - self.prime.position)**2) < self.sc.UNIT_DOWN_RAD**2:
                self.prime.took_down = True
        #crash (invader and prime)
        for i in self.invaders:
            if not i.captured and np.sum((i.position - self.prime.position)**2) < self.sc.UNIT_DOWN_RAD**2:
                self.prime.took_down = True

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
            self.sc.u_dot.set_3d_properties([p.position[2]])  
        pos_arr_u = np.array(self.hist_prime)
        self.sc.u_path.set_data(pos_arr_u[:, 0], pos_arr_u[:, 1])
        if self._3d:
            self.sc.u_path.set_3d_properties(pos_arr_u[:,2])
        #ellipse
        if not self._3d:
            self._update_ellipse()
        #if all invaders are captured, or prime unit was taken down or has finished, the animation will end
        # if (self.captured_count >= sc.INVADER_NUM and self.prime.finished) or self.prime.took_down: # or state["prime"].finished:
        #     self.anim.event_source.stop()
        if self._3d:
            return self.sc.p_dots + self.sc.i_dots + self.sc.p_paths + self.sc.i_paths + \
                [self.sc.u_dot, self.sc.u_path]
        else:
            return self.sc.p_dots + self.sc.i_dots + self.sc.p_paths + self.sc.i_paths + \
                [self.sc.u_dot, self.sc.u_path, self.ellipse_patch]

    def run(self):
        #running the animation
        self.anim = FuncAnimation(self.fig, self.update, frames=200, interval=50, blit=True)
        plt.show()
