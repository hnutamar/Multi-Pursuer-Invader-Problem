import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from pursuer import Pursuer
from invader import Invader
from matplotlib.animation import FuncAnimation
import sim_config as sc

x_border = sc.WORLD_WIDTH/4
y_border = sc.WORLD_HEIGHT/4
rnd_points = np.random.uniform(low=[-1 + x_border, -1 + y_border], high=[sc.WORLD_WIDTH - x_border, sc.WORLD_HEIGHT - y_border], size=(sc.PURSUER_NUM + sc.INVADER_NUM, 2))

pursuers = []
positions_p = [[] for _ in range(sc.PURSUER_NUM)]
for i in range(0, sc.PURSUER_NUM):
    pursuers.append(Pursuer(position=rnd_points[i], speed=0.7))
    positions_p[i].append(rnd_points[i])
    
invader = []
positions_i = []
for i in range(sc.INVADER_NUM):
    invader.append(Invader(position=rnd_points[i + sc.PURSUER_NUM], speed=1.2))
    positions_i.append(rnd_points[i + sc.PURSUER_NUM])

#animation
def update(frame):
    dirs_p = []
    for pursuer in pursuers:
        dirs_p.append(pursuer.pursue(invader[0]))
    dir_i = invader[0].evade(pursuer)

    for pursuer, dir in zip(pursuers, dirs_p):
        pursuer.move(dir)
    invader[0].move(dir_i)

    for pursuer, pos in zip(pursuers, positions_p):
        pos.append(pursuer.position.copy())
    positions_i.append(invader[0].position.copy())

    for dot, pursuer in zip(sc.p_dots, pursuers):
        dot.set_data([pursuer.position[0]], [pursuer.position[1]])
    sc.i_dot.set_data([invader[0].position[0]], [invader[0].position[1]])
    
    for path, pos in zip(sc.p_paths, positions_p):
        path.set_data([p[0] for p in pos], [p[1] for p in pos])
    sc.i_path.set_data([i[0] for i in positions_i], [i[1] for i in positions_i])

    #capture check (if pursuer is close enough to invader)
    for pursuer in pursuers:
        if np.sum((pursuer.position - invader[0].position)**2) < sc.CAPTURE_RAD**2:
            anim.event_source.stop()
            break

    return sc.p_dots + [sc.i_dot] + sc.p_paths + [sc.i_path]

anim = FuncAnimation(sc.fig, update, frames=200, interval=50, blit=True)
plt.show()
