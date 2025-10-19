import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from pursuer import Pursuer
from invader import Invader
from prime_unit import Prime_unit
from matplotlib.animation import FuncAnimation
import sim_config as sc
from scipy.optimize import linear_sum_assignment

def formation_calculator(purs: list[Pursuer], unit: Prime_unit):
    formation_r = max(4*purs[0].dist_formation/(np.pi*2), purs[0].min_formation_r)
    n = len(pursuers)
    angle_piece = 2*np.pi/n
    angles = np.arange(n)*angle_piece
    cx, cy = unit.position
    form_pos = np.stack([cx + formation_r * np.cos(angles), cy + formation_r * np.sin(angles)], axis=1)
    pos_now = np.array([pur.position for pur in purs])
    diff = pos_now[:, np.newaxis, :] - form_pos[np.newaxis, :, :]
    D = np.linalg.norm(diff, axis=2)
    row_idx, col_idx = linear_sum_assignment(D)
    for i in range(n):
        purs[i].num = col_idx[i]

x_border = sc.WORLD_WIDTH/6
y_border = sc.WORLD_HEIGHT/6
#prime unit init
pos_u = [3.0, 3.0]
prime_unit = Prime_unit(position=pos_u, speed=0.08, max_omega=1.0)
positions_u = [pos_u]
way_point = np.array([sc.WORLD_WIDTH - x_border, sc.WORLD_HEIGHT - y_border])
#random inital pos
rnd_points_purs = np.random.uniform(low=[-sc.PURSUER_NUM/2 - 2 + pos_u[0], -sc.PURSUER_NUM/2 - 2 + pos_u[1]], high=[sc.PURSUER_NUM/2 + 2 + pos_u[0], sc.PURSUER_NUM/2 + 2 + pos_u[1]], size=(sc.PURSUER_NUM, 2))
rnd_points_inv = np.random.uniform(low=[-1 + x_border, -1 + y_border], high=[sc.WORLD_WIDTH - x_border, sc.WORLD_HEIGHT - y_border], size=(sc.INVADER_NUM, 2))
#pursuers init
pursuers = []
positions_p = [[] for _ in range(sc.PURSUER_NUM)]
for i in range(sc.PURSUER_NUM):
    pursuers.append(Pursuer(position=rnd_points_purs[i], speed=0.3, max_omega=1.5, num=i))
    positions_p[i].append(rnd_points_purs[i])
formation_calculator(pursuers, prime_unit)
#invaders init
invaders = []
positions_i = [[] for _ in range(sc.INVADER_NUM)]
for i in range(sc.INVADER_NUM):
    invaders.append(Invader(position=rnd_points_inv[i], speed=0.2, max_omega=1.5))
    positions_i[i].append(rnd_points_inv[i])
#invader captures counter
invader_captured = [0]

#animation
def update(frame):
    #makes np arrays of positions of still not captured invaders and all pursuers
    free_invaders = [inv for inv in invaders if not inv.captured]
    not_crashed_pursuers = [pur for pur in pursuers if not pur.crashed]
    #pursuers_pos = np.array([pur.position for pur in pursuers])
    #new directions of all drones
    dirs_i = [invader.evade(not_crashed_pursuers, prime_unit) for invader in invaders]
    dirs_p = [pursuer.pursue(free_invaders, not_crashed_pursuers, prime_unit) for pursuer in pursuers]
    dir_u = prime_unit.fly(way_point)
    #making the move in that dir according to the time and speed
    for pursuer, dir in zip(pursuers, dirs_p):
        pursuer.move(dir)
    for invader, dir in zip(invaders, dirs_i):
        invader.move(dir)
    prime_unit.move(dir_u)
    #positions for animations
    for pursuer, pos in zip(pursuers, positions_p):
        pos.append(pursuer.position.copy())
    for invader, pos in zip(invaders, positions_i):
        pos.append(invader.position.copy())
    positions_u.append(prime_unit.position.copy())
    #dots representing the current positions of drones
    for dot, pursuer in zip(sc.p_dots, pursuers):
        dot.set_data([pursuer.position[0]], [pursuer.position[1]])
    for dot, invader in zip(sc.i_dots, invaders):
        dot.set_data([invader.position[0]], [invader.position[1]])
    sc.u_dot.set_data([prime_unit.position[0]], [prime_unit.position[1]])
    #the whole path of all the drones is needed for animation
    for path, pos in zip(sc.p_paths, positions_p):
        pos_arr = np.array(pos)
        path.set_data(pos_arr[:,0], pos_arr[:,1])
    for path, pos in zip(sc.i_paths, positions_i):
        pos_arr = np.array(pos)
        path.set_data(pos_arr[:,0], pos_arr[:,1])
    pos_arr = np.array(positions_u)
    sc.u_path.set_data(pos_arr[:,0], pos_arr[:,1])
    
    free_invaders = [inv for inv in invaders if not inv.captured]
    for pursuer in pursuers:
        #capture check (if pursuer is close enough to invader)
        for invader in free_invaders:
            if np.sum((pursuer.position - invader.position)**2) < sc.CAPTURE_RAD**2 and invader.captured == False:
                invader_captured[0] += 1
                invader.captured = True
        for purs in pursuers:
            if not (purs is pursuer) and np.sum((pursuer.position - purs.position)**2) < sc.CRASH_RAD**2 and purs.crashed == False and pursuer.crashed == False:
                pursuer.crashed = True
                purs.crashed = True
        #crash to prime_unit check
        if np.sum((pursuer.position - prime_unit.position)**2) < sc.UNIT_DOWN_RAD**2 and pursuer.crashed == False:
            prime_unit.took_down = True
            break
    #crash to prime_unit check        
    for invader in invaders:
        if np.sum((invader.position - prime_unit.position)**2) < sc.UNIT_DOWN_RAD**2 and invader.captured == False:
            prime_unit.took_down = True
            break
    #if all invaders are captured, the animation will end
    if (invader_captured[0] >= sc.INVADER_NUM and prime_unit.finished) or prime_unit.took_down or prime_unit.finished:
        anim.event_source.stop()
    #returning paths and positions of all drones for animation
    return sc.p_dots + sc.i_dots + sc.p_paths + sc.i_paths + [sc.u_dot] + [sc.u_path]

anim = FuncAnimation(sc.fig, update, frames=200, interval=50, blit=True)
plt.show()
