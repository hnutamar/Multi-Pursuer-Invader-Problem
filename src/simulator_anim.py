import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from pursuer import Pursuer
from invader import Invader
from prime_unit import Prime_unit
from matplotlib.animation import FuncAnimation
import sim_config as sc

x_border = sc.WORLD_WIDTH/6
y_border = sc.WORLD_HEIGHT/6
#random inital pos
rnd_points = np.random.uniform(low=[-1 + x_border, -1 + y_border], high=[sc.WORLD_WIDTH - x_border, sc.WORLD_HEIGHT - y_border], size=(sc.PURSUER_NUM + sc.INVADER_NUM, 2))
#pursuers init
pursuers = []
positions_p = [[] for _ in range(sc.PURSUER_NUM)]
for i in range(sc.PURSUER_NUM):
    pursuers.append(Pursuer(position=rnd_points[i], speed=0.8))
    positions_p[i].append(rnd_points[i])
#invaders init
invaders = []
positions_i = [[] for _ in range(sc.INVADER_NUM)]
for i in range(sc.INVADER_NUM):
    invaders.append(Invader(position=rnd_points[i + sc.PURSUER_NUM], speed=1.2))
    positions_i[i].append(rnd_points[i + sc.PURSUER_NUM])

prime_unit = Prime_unit(position=[0, 0], speed=0.3)
positions_u = [[0, 0]]
way_point = np.array([sc.WORLD_WIDTH - x_border, sc.WORLD_HEIGHT - y_border])
#invader captures counter
invader_captured = [0]

#animation
def update(frame):
    #makes np arrays of positions of still not captured invaders and all pursuers
    free_invaders_pos = np.array([inv.position for inv in invaders if not inv.captured])
    pursuers_pos = np.array([pur.position for pur in pursuers])
    #new directions of all drones
    dirs_p = [pursuer.pursue(free_invaders_pos) for pursuer in pursuers]
    dirs_i = [invader.evade(pursuers_pos) for invader in invaders]
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
    #capture check (if pursuer is close enough to invader)
    free_invaders = [inv for inv in invaders if not inv.captured]
    for pursuer in pursuers:
        for invader in free_invaders:
            if np.sum((pursuer.position - invader.position)**2) < sc.CAPTURE_RAD**2 and invader.captured == False:
                invader_captured[0] += 1
                invader.captured = True
    #if all invaders are captured, the animation will end
    if invader_captured[0] >= sc.INVADER_NUM and prime_unit.finished:
        anim.event_source.stop()
    #returning paths and positions of all drones for animation
    return sc.p_dots + sc.i_dots + sc.p_paths + sc.i_paths + [sc.u_dot] + [sc.u_path]

anim = FuncAnimation(sc.fig, update, frames=200, interval=50, blit=True)
plt.show()
