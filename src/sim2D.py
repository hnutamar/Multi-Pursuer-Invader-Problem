import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from pursuer import Pursuer
from invader import Invader
from prime_unit import Prime_unit
from matplotlib.animation import FuncAnimation
import sim_config2D as sc
from scipy.optimize import linear_sum_assignment
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

def ellipse_change(ellipse, unit):
    ellipse.center = unit.center
    ellipse.width = unit.axis_a*2
    ellipse.height = unit.axis_b*2
    ellipse.angle = unit.rot_angle*(180/np.pi)
    return

#state of the drones
state = {"form_count": [],#not used right now
         "pursuers": [],
         "invaders": [],
         "prime": None,
         "inv_captured": 0}
x_border = sc.WORLD_WIDTH/6
y_border = sc.WORLD_HEIGHT/6
#prime unit init
pos_u = [3.0, 3.0]
state["prime"] = Prime_unit(position=pos_u, max_acc=0.7, max_omega=1.0)
positions_u = [pos_u]
way_point = np.array([sc.WORLD_WIDTH/2, sc.WORLD_HEIGHT/2])
#way_point = np.array([sc.WORLD_WIDTH - x_border, sc.WORLD_HEIGHT - y_border])
#random inital pos
rnd_points_purs = np.random.uniform(low=[-sc.PURSUER_NUM/2 - 2 + pos_u[0], -sc.PURSUER_NUM/2 - 2 + pos_u[1]], high=[sc.PURSUER_NUM/2 + 2 + pos_u[0], sc.PURSUER_NUM/2 + 2 + pos_u[1]], size=(sc.PURSUER_NUM, 2))
rnd_points_inv = np.random.uniform(low=[sc.PURSUER_NUM/2 + 2 + pos_u[0], sc.PURSUER_NUM/2 + 2 + pos_u[1]], high=[sc.WORLD_WIDTH - x_border, sc.WORLD_HEIGHT - y_border], size=(sc.INVADER_NUM, 2))
rnd_acc_inv = np.random.uniform(low=0.9, high=1.7, size=(sc.INVADER_NUM,))
#pursuers init
state["pursuers"] = []
positions_p = [[] for _ in range(sc.PURSUER_NUM)]
for i in range(sc.PURSUER_NUM):
    state["pursuers"].append(Pursuer(position=rnd_points_purs[i], max_acc=2.5, max_omega=1.5, num=i, purs_num=sc.PURSUER_NUM))
    positions_p[i].append(rnd_points_purs[i])
#invaders init
state["invaders"] = []
positions_i = [[] for _ in range(sc.INVADER_NUM)]
for i in range(sc.INVADER_NUM):
    state["invaders"].append(Invader(position=rnd_points_inv[i], max_acc=rnd_acc_inv[i], max_omega=1.5))
    positions_i[i].append(rnd_points_inv[i])
#how many pursuers are in formation
#state["form_count"] = [pur for pur in state["pursuers"] if pur.state == States.FORM]

ellipse = Ellipse(state["prime"].center, state["prime"].axis_a*2, state["prime"].axis_b*2, angle=state["prime"].rot_angle*(180/np.pi),
                  edgecolor="#10ec22", facecolor='none', lw=0.5, linestyle='--')
sc.ax.add_patch(ellipse)

#animation
def update(frame):
    #check pursuers states
    # form_max = state["pursuers"][0].form_max[0]
    # form_now = [pur for pur in state["pursuers"] if (pur.state == States.FORM and np.linalg.norm(pur.position - state["prime"].position) < form_max)]
    # if len(state["form_count"]) != len(form_now):
    #     state["form_count"] = form_now
    #     formation_calculator(state["pursuers"], state["prime"], form_max)
    # elif not all(x is y for x, y in zip(state["form_count"], form_now)):
    #     state["form_count"] = form_now
    #     #print("1")
    #     formation_calculator(state["pursuers"], state["prime"], form_max)
    #makes np arrays of positions of still not captured invaders and all pursuers
    free_inv = [inv for inv in state["invaders"] if not inv.captured]
    free_purs = [pur for pur in state["pursuers"] if pur.state != States.CRASHED]
    #pursuers_pos = np.array([pur.position for pur in pursuers])
    #new directions of all drones
    dirs_i = [invader.evade(free_purs, state["prime"]) for invader in state["invaders"]]
    dirs_p = []
    for purs in state["pursuers"]:
        close_purs = [p for p in free_purs if np.linalg.norm(p.position - purs.position) <= purs.vis_r]
        dirs_p.append(purs.pursue(free_inv, close_purs, state["prime"]))
    #dirs_p = [pursuer.pursue(free_inv, free_purs, state["prime"]) for pursuer in state["pursuers"]]
    dir_u = state["prime"].fly(way_point, free_inv, free_purs)
    #making the move in that dir according to the time and speed
    for p, p_dir in zip(state["pursuers"], dirs_p):
        p.move(p_dir)
    for i, i_dir in zip(state["invaders"], dirs_i):
        i.move(i_dir)
    state["prime"].move(dir_u)
    #positions for animations
    for p, p_pos in zip(state["pursuers"], positions_p):
        p_pos.append(p.position.copy())
    for i, i_pos in zip(state["invaders"], positions_i):
        i_pos.append(i.position.copy())
    positions_u.append(state["prime"].position.copy())
    #dots representing the current positions of drones
    for p_dot, p in zip(sc.p_dots, state["pursuers"]):
        p_dot.set_data([p.position[0]], [p.position[1]])
        if p.state == States.PURSUE:
            p_dot.set_color("#631616")
        else:
            p_dot.set_color('#d62728')
    for i_dot, i in zip(sc.i_dots, state["invaders"]):
        i_dot.set_data([i.position[0]], [i.position[1]])
    sc.u_dot.set_data([state["prime"].position[0]], [state["prime"].position[1]])
    #the whole path of all the drones is needed for animation
    for p_path, pos in zip(sc.p_paths, positions_p):
        pos_arr = np.array(pos)
        p_path.set_data(pos_arr[:,0], pos_arr[:,1])
    for i_path, pos in zip(sc.i_paths, positions_i):
        pos_arr = np.array(pos)
        i_path.set_data(pos_arr[:,0], pos_arr[:,1])
    pos_arr = np.array(positions_u)
    sc.u_path.set_data(pos_arr[:,0], pos_arr[:,1])
    #colision check
    for p in free_purs:
        #capture check (if pursuer is close enough to invader)
        for i in free_inv:
            if np.sum((p.position - i.position)**2) < sc.CAPTURE_RAD**2 and i.captured == False:
                state["inv_captured"] += 1
                i.captured = True
                if i.pursuer is not None:
                    i.pursuer.target = None
        #crash to other pursuers check
        for other in free_purs:
            if not (other is p) and np.sum((p.position - other.position)**2) < sc.CRASH_RAD**2: #and other.state != States.CRASHED and p.state != States.CRASHED:
                p.state = States.CRASHED
                other.state = States.CRASHED
        #crash to prime_unit check
        if np.sum((p.position - state["prime"].position)**2) < sc.UNIT_DOWN_RAD**2: #p.state != States.CRASHED
            state["prime"].took_down = True
            break
    #crash to prime_unit check        
    for i in state["invaders"]:
        if i.captured == False and np.sum((i.position - state["prime"].position)**2) < sc.UNIT_DOWN_RAD**2:
            state["prime"].took_down = True
            break
    
    ellipse_change(ellipse, state["prime"])
    #if all invaders are captured, or prime unit was taken down or has finished, the animation will end
    #if (state["inv_captured"] >= sc.INVADER_NUM and state["prime"].finished) or state["prime"].took_down: # or state["prime"].finished:
    #    anim.event_source.stop()
    #returning paths and positions of all drones for animation
    return sc.p_dots + sc.i_dots + sc.p_paths + sc.i_paths + [sc.u_dot] + [sc.u_path] + [ellipse]

anim = FuncAnimation(sc.fig, update, frames=200, interval=50, blit=True)
plt.show()
