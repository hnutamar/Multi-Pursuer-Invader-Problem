import numpy as np
from pursuer import Pursuer
from invader import Invader
from prime_unit import Prime_unit
from pursuer_states import States
from prime_mode import Modes

class SimulationWorld:
    def __init__(self, sc_config, _3d=False, purs_acc=None, prime_acc=None, inv_acc=None, 
                 prime_pos=None, inv_pos=None, purs_pos=None, purs_num=None):
        self.sc = sc_config
        self._3d = _3d
        #parameters for reset
        self.init_params = {
            'purs_acc': purs_acc, 'prime_acc': prime_acc, 'inv_acc': inv_acc,
            'prime_pos': prime_pos, 'inv_pos': inv_pos, 'purs_pos': purs_pos, 'purs_num': purs_num
        }
        self.reset()

    def reset(self):
        #resets to initial state
        self.time = 0.0
        self.captured_count = 0
        self.purs_crash = False
        self.prime_crashed = False
        #inits of agents
        self._init_agents(**self.init_params)
        return self.get_state()

    def _init_agents(self, purs_acc, prime_acc, inv_acc, prime_pos, inv_pos, purs_pos, purs_num):
        #obstacle
        self.obstacle = None
        if not self._3d and self.sc.obs_patch is not None:
             #only physics
            self.obstacle = type('obj', (object,), {'center': self.sc.obs_patch.center, 'radius': self.sc.obs_patch.radius})
        elif self._3d and self.sc.obs_patch is not None:
            self.obstacle = []
            for i in range(len(self.sc.obs_patch)):
                obs_pos = self.sc.obs_pos[i]
                obs_rad = self.sc.obs_rads[i]
                obstacle = {'center': obs_pos, 'radius': obs_rad}
                self.obstacle.append(obstacle)
            print(self.obstacle)
            print(len(self.sc.obs_patch))
        #borders and waypoints
        x_border = self.sc.WORLD_WIDTH / 6
        y_border = self.sc.WORLD_HEIGHT / 6
        if self._3d:
            z_border = self.sc.WORLD_Z / 6
            self.way_point = np.array([self.sc.WORLD_WIDTH - x_border, self.sc.WORLD_HEIGHT - y_border, self.sc.WORLD_Z - z_border])
        else:
            #only line
            self.way_point = np.array([self.sc.WORLD_WIDTH - x_border, self.sc.WORLD_HEIGHT - y_border])
        #Prime
        acc_prime = prime_acc or 0.1
        if self._3d:
            pos_u = prime_pos if prime_pos is not None else np.array([3.0, 3.0, 3.0])
        else:
            pos_u = prime_pos if prime_pos is not None else np.array([3.0, 3.0])
            
        self.prime = Prime_unit(position=pos_u, max_acc=acc_prime, max_omega=1.0, my_rad=self.sc.UNIT_RAD)
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
                #low=[self.sc.PURSUER_NUM/2 + 2 + pos_u[0], self.sc.PURSUER_NUM/2 + 2 + pos_u[1]],  
                low=[0.0, self.sc.PURSUER_NUM/2 + 2 + pos_u[1]],  
                high=[self.sc.WORLD_WIDTH - x_border, self.sc.WORLD_HEIGHT - y_border], 
                size=(self.sc.INVADER_NUM, 2)
            )
        rnd_acc_inv = np.full(self.sc.INVADER_NUM, inv_acc) if inv_acc is not None else np.random.uniform(1.3, 1.5, self.sc.INVADER_NUM)
        acc_purs = purs_acc or 1.0
        #pursuer init
        self.pursuers = []
        for i in range(self.sc.PURSUER_NUM):
            p_num = purs_num[i] if purs_num is not None else np.random.randint(0, 1001)
            p = Pursuer(position=rnd_points_purs[i], max_acc=acc_purs, max_omega=1.5, my_rad=self.sc.DRONE_RAD, purs_num=p_num)
            self.pursuers.append(p)
        #invader init
        self.invaders = []
        for i in range(self.sc.INVADER_NUM):
            inv = Invader(position=rnd_points_inv[i], max_acc=rnd_acc_inv[i], max_omega=1.5, my_rad=self.sc.DRONE_RAD)
            self.invaders.append(inv)

    def step(self, dt=0.1, manual_invader_vel=None):
        #filtering living drones
        free_inv = [inv for inv in self.invaders if not inv.crashed]
        free_purs = [pur for pur in self.pursuers if not pur.crashed]
        #invader acc
        dirs_i = []
        for idx, inv in enumerate(self.invaders):
            #manual control
            if idx == 0 and not inv.crashed and manual_invader_vel is not None:
                acc_size = np.linalg.norm(manual_invader_vel)
                if acc_size > 0:
                    inv_acc = (manual_invader_vel / acc_size) * inv.max_acc
                else:
                    inv_acc = np.zeros_like(manual_invader_vel)
                dirs_i.append(inv_acc)
            else:
                # AI logic
                dirs_i.append(inv.evade(free_purs, self.prime, self.obstacle))
        #pursuer acc
        dirs_p = []
        for purs in self.pursuers:
            close_purs = [p for p in free_purs if (np.linalg.norm(p.position - purs.position) - p.my_rad - purs.my_rad) <= self.sc.PURS_VIS]
            dirs_p.append(purs.pursue(free_inv, close_purs, self.prime, self.obstacle))
        #prime acc
        dir_u = self.prime.fly(self.way_point, free_inv, free_purs, Modes.LINE)
        #drones moving
        for p, p_dir in zip(self.pursuers, dirs_p):
            p.move(p_dir)
        for i, i_dir in zip(self.invaders, dirs_i):
            i.move(i_dir)
        self.prime.move(dir_u)
        #counter
        self.time += dt
        #COLLISIONS
        #Prime and Invader
        for i in self.invaders:
            if not i.crashed and np.sum((i.position - self.prime.position)**2) < self.sc.UNIT_DOWN_RAD**2:
                self.prime.crashed = True
            i.purs_num = 0
        #Prime and Obstacle
        if self.obstacle is not None:
            for obstacle in self.obstacle:
                if not self._3d and np.sum((self.prime.position - self.obstacle.center)**2) < self.obstacle.radius**2:
                    self.prime.crashed = True
                    break
                elif self._3d and np.sum((self.prime.position - obstacle['center'])**2) < obstacle['radius']**2:
                    self.prime.crashed = True
                    break
        #collisions and pursue check
        for p in free_purs:
            #pursue check
            if p.target is not None:
                p.target[0].purs_num += 1
            #obstacle check
            if self.sc.obs_patch is not None:
                for obstacle in self.obstacle:
                    if not self._3d and np.sum((p.position - self.sc.obs_patch.center)**2) < self.sc.obs_patch.radius**2:
                        p.crashed = True
                        break
                    elif self._3d and np.sum((p.position - obstacle['center'])**2) < obstacle['radius']**2:
                        p.crashed = True
                        break
            #capture check
            for i in free_inv:
                if np.sum((p.position - i.position)**2) < self.sc.CAPTURE_RAD**2 and not i.crashed:
                    self.captured_count += 1
                    i.crashed = True
            #Pursuer and Pursuer
            for other in free_purs:
                if (other is not p) and np.sum((p.position - other.position)**2) < self.sc.CRASH_RAD**2:
                    p.crashed = True
                    other.crashed = True
            #Pursuer and Prime
            if np.sum((p.position - self.prime.position)**2) < self.sc.UNIT_DOWN_RAD**2:
                self.prime.crashed = True
        #ending check
        done = self.prime.crashed or self.prime.finished #or (self.captured_count == self.sc.INVADER_NUM) 
        #reward for RL
        reward = 0
        return self.get_state(), reward, done

    def get_state(self):
        #returns state of the simulation
        return {
            "prime": self.prime.position.copy(),
            "pursuers": [p.position.copy() for p in self.pursuers],
            "pursuers_status": [[p.state, p.target] for p in self.pursuers], #for colors
            "invaders": [i.position.copy() for i in self.invaders],
            "time": self.time
        }