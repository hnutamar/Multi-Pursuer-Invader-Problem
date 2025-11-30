import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from pursuer import Pursuer
from invader import Invader
from prime_unit import Prime_unit
from matplotlib.animation import FuncAnimation
import sim_config3D as sc
from scipy.optimize import linear_sum_assignment
from pursuer_states import States
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

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
z_border = sc.WORLD_Z/6
#prime unit init
pos_u = [3.0, 3.0, 3.0]
state["prime"] = Prime_unit(position=pos_u, max_acc=0.7, max_omega=1.0)
positions_u = [pos_u]
#way_point = np.array([sc.WORLD_WIDTH/2, sc.WORLD_HEIGHT/2, 7.0])
way_point = np.array([sc.WORLD_WIDTH - x_border, sc.WORLD_HEIGHT - y_border, sc.WORLD_Z - z_border])
#random inital pos
rnd_points_purs = np.random.uniform(low=[-sc.PURSUER_NUM/2 - 2 + pos_u[0], -sc.PURSUER_NUM/2 - 2 + pos_u[1], pos_u[2]], high=[sc.PURSUER_NUM/2 + 2 + pos_u[0], sc.PURSUER_NUM/2 + 2 + pos_u[1], pos_u[2]], size=(sc.PURSUER_NUM, 3))
rnd_points_inv = np.random.uniform(low=[sc.PURSUER_NUM/2 + 2 + pos_u[0], sc.PURSUER_NUM/2 + 2 + pos_u[1], pos_u[2]], high=[sc.WORLD_WIDTH - x_border, sc.WORLD_HEIGHT - y_border, pos_u[2]], size=(sc.INVADER_NUM, 3))
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

# ellipse = Ellipse(state["prime"].center, state["prime"].axis_a*2, state["prime"].axis_b*2, angle=state["prime"].rot_angle*(180/np.pi),
#                   edgecolor="#10ec22", facecolor='none', lw=0.5, linestyle='--')
# sc.ax.add_patch(ellipse)

#animation
def update(frame):
    #makes np arrays of positions of still not captured invaders and all pursuers
    free_inv = [inv for inv in state["invaders"] if not inv.captured]
    free_purs = [pur for pur in state["pursuers"] if pur.state != States.CRASHED]
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
        p_dot.set_3d_properties([p.position[2]])
        if p.state == States.PURSUE:
            p_dot.set_color("#631616")
        else:
            p_dot.set_color('#d62728')
    for i_dot, i in zip(sc.i_dots, state["invaders"]):
        i_dot.set_data([i.position[0]], [i.position[1]])
        i_dot.set_3d_properties([i.position[2]])
    sc.u_dot.set_data([state["prime"].position[0]], [state["prime"].position[1]])
    sc.u_dot.set_3d_properties([state["prime"].position[2]])
    #the whole path of all the drones is needed for animation
    for p_path, pos in zip(sc.p_paths, positions_p):
        pos_arr = np.array(pos)
        if len(pos_arr) > 0:
            p_path.set_data(pos_arr[:,0], pos_arr[:,1])
            p_path.set_3d_properties(pos_arr[:,2])
    for i_path, pos in zip(sc.i_paths, positions_i):
        pos_arr = np.array(pos)
        if len(pos_arr) > 0:
            i_path.set_data(pos_arr[:,0], pos_arr[:,1])
            i_path.set_3d_properties(pos_arr[:,2])
    pos_arr = np.array(positions_u)
    if len(pos_arr) > 0:
        sc.u_path.set_data(pos_arr[:,0], pos_arr[:,1])
        sc.u_path.set_3d_properties(pos_arr[:,2])
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
    
    #ellipse_change(ellipse, state["prime"])
    #if all invaders are captured, or prime unit was taken down or has finished, the animation will end
    #if (state["inv_captured"] >= sc.INVADER_NUM and state["prime"].finished) or state["prime"].took_down: # or state["prime"].finished:
    #    anim.event_source.stop()
    #returning paths and positions of all drones for animation
    return sc.p_dots + sc.i_dots + sc.p_paths + sc.i_paths + [sc.u_dot] + [sc.u_path] #+ [ellipse]

anim = FuncAnimation(sc.fig, update, frames=200, interval=50, blit=True)
plt.show()
