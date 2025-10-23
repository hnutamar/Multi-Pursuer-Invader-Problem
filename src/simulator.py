import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
from pursuer import Pursuer
from invader import Invader

# Inicializace
pursuer = Pursuer(position=[0, 0], max_acc=1.2)
invader = Invader(position=[5, 5], max_acc=1.0)

dt = 0.1
positions_p = [pursuer.position.copy()]
positions_i = [invader.position.copy()]

for t in np.arange(0, 20, dt):
    dir_p = pursuer.pursue(invader)
    dir_i = invader.evade(pursuer)

    pursuer.move(dir_p, dt)
    invader.move(dir_i, dt)

    positions_p.append(pursuer.position.copy())
    positions_i.append(invader.position.copy())

# Vizualizace trajektorií
positions_p = np.array(positions_p)
positions_i = np.array(positions_i)

plt.plot(positions_p[:, 0], positions_p[:, 1], label='Pursuer', color='red')
plt.plot(positions_i[:, 0], positions_i[:, 1], label='Invader', color='blue')
plt.scatter(*positions_p[0], color='red', marker='o')
plt.scatter(*positions_i[0], color='blue', marker='x')
plt.legend()
plt.axis('equal')
plt.show()
