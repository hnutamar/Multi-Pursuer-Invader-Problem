import numpy as np
from pursuer import Pursuer
from invader import Invader
from prime_unit import Prime_unit
from pursuer_states import States
from prime_mode import Modes
from scipy.spatial.distance import cdist

class SimulationWorld:
    def __init__(self, sc_config, _3d=False, purs_acc=None, purs_speed=None, prime_acc=None, prime_speed=None, inv_acc=None, inv_speed=None,
                 prime_pos=None, inv_pos=None, purs_pos=None, purs_num=None):
        self.sc = sc_config
        self._3d = _3d
        #parameters for reset
        self.init_params = {
            'purs_acc': purs_acc, 'purs_speed': purs_speed, 'prime_acc': prime_acc, 'prime_speed': prime_speed, 'inv_acc': inv_acc, 'inv_speed': inv_speed,
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

    def _init_agents(self, purs_acc, purs_speed, prime_acc, prime_speed, inv_acc, inv_speed, prime_pos, inv_pos, purs_pos, purs_num):
        #obstacle
        self.obs_centers = None
        self.obs_radii = None
        if self.sc.obstacle:
            self.obs_centers = np.array(self.sc.obs_pos, dtype=float)
            self.obs_radii = np.array(self.sc.obs_rads, dtype=float)
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
        speed_prime = prime_speed or 0.5
        acc_prime = prime_acc or 0.1
        if self._3d:
            pos_u = prime_pos if prime_pos is not None else np.array([3.0, 3.0, 7.0])
        else:
            pos_u = prime_pos if prime_pos is not None else np.array([3.0, 3.0])
            
        self.prime = Prime_unit(position=pos_u, max_speed=speed_prime, max_acc=acc_prime, max_omega=1.0, my_rad=self.sc.UNIT_RAD, dt=self.sc.DT)
        #init positions and acceleration of agents (random)
        if self._3d:
            rnd_points_purs = purs_pos if purs_pos is not None else np.random.uniform(
                low=[-self.sc.PURSUER_NUM/2 - 2 + pos_u[0], -self.sc.PURSUER_NUM/2 - 2 + pos_u[1], pos_u[2]], 
                high=[self.sc.PURSUER_NUM/2 + 2 + pos_u[0], self.sc.PURSUER_NUM/2 + 2 + pos_u[1], self.sc.PURSUER_NUM/2 + 2 +  pos_u[2]], 
                size=(self.sc.PURSUER_NUM, 3)
            )
            rnd_points_inv = inv_pos if inv_pos is not None else np.random.uniform(
                low=[-2*self.sc.WORLD_WIDTH, -2*self.sc.WORLD_HEIGHT, pos_u[2]], 
                high=[2*self.sc.WORLD_WIDTH, 2*self.sc.WORLD_HEIGHT, pos_u[2] + 15], 
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
                low=[0, 0],  
                high=[2*self.sc.WORLD_WIDTH, 2*self.sc.WORLD_HEIGHT], 
                size=(self.sc.INVADER_NUM, 2)
            )
        rnd_acc_inv = np.full(self.sc.INVADER_NUM, inv_acc) if inv_acc is not None else np.random.uniform(1.3, 1.5, self.sc.INVADER_NUM)
        acc_inv = inv_acc or 1.0
        speed_inv = inv_speed or 5.0
        acc_purs = purs_acc or 1.0
        speed_purs = purs_speed or 5.0
        #pursuer init
        self.pursuers = []
        for i in range(self.sc.PURSUER_NUM):
            p_num = purs_num[i] if purs_num is not None else np.random.randint(0, 1001)
            p = Pursuer(position=rnd_points_purs[i], max_speed=speed_purs,
                max_acc=acc_purs, max_omega=1.5, my_rad=self.sc.DRONE_RAD, purs_num=p_num, purs_vis=self.sc.PURS_VIS, dt=self.sc.DT)
            self.pursuers.append(p)
        #invader init
        self.invaders = []
        for i in range(self.sc.INVADER_NUM):
            inv = Invader(position=rnd_points_inv[i], max_speed=speed_inv, max_acc=acc_inv, max_omega=1.5, my_rad=self.sc.DRONE_RAD, dt=self.sc.DT)
            self.invaders.append(inv)

    def _get_safe_agent_data(self, agents):
        dim = 3 if self._3d else 2
        #no one is alive
        if not agents:
            return (np.empty((0, dim)), 
                    np.empty((0,)), 
                    np.empty((0,)))
        #living
        pos = np.vstack([a.position for a in agents])
        rad = np.array([a.my_rad for a in agents])
        #in case of invaders
        try:
            p_nums = np.array([a.purs_num for a in agents])
        except AttributeError:
            p_nums = np.zeros(len(agents)) #fallback    
        return pos, rad, p_nums

    def step(self, manual_invader_vel=None):
        #counter
        self.time += self.sc.DT
        #filtering living drones
        free_inv = [inv for inv in self.invaders if not inv.crashed]
        free_purs = [pur for pur in self.pursuers if not pur.crashed]
        all_inv_pos, all_inv_rad, all_inv_pnums = self._get_safe_agent_data(free_inv)
        all_purs_pos, all_purs_rad, _ = self._get_safe_agent_data(free_purs)
        #invader move
        for i, inv in enumerate(self.invaders):
            #manual control
            if i == 0 and not inv.crashed and manual_invader_vel is not None:
                acc_size = np.linalg.norm(manual_invader_vel)
                if acc_size > 0:
                    inv_acc = (manual_invader_vel / acc_size) * inv.max_acc
                else:
                    inv_acc = np.zeros_like(manual_invader_vel)
            else:
                # AI logic
                inv_acc = inv.evade(free_purs, self.prime, (self.obs_centers, self.obs_radii))
            inv.move(inv_acc)
        #pursuer move
        for i, pur in enumerate(free_purs):
            close_purs = [p for p in free_purs if (np.linalg.norm(p.position - pur.position) - p.my_rad - pur.my_rad) <= self.sc.PURS_VIS]
            purs_acc = pur.pursue(free_inv, close_purs, self.prime, 
                       precalc_data=(all_inv_pos, all_inv_pnums, all_inv_rad, all_purs_pos, all_purs_rad, i, self.obs_centers, self.obs_radii))
            pur.move(purs_acc)
        #prime move
        prime_acc = self.prime.fly(self.way_point, free_inv, free_purs, Modes.LINE, (self.obs_centers, self.obs_radii))
        self.prime.move(prime_acc)
        #number of pursuers pursuing invaders
        for i in free_inv:
            i.purs_num = 0
        for p in free_purs:
            if p.target is not None:
                p.target[0].purs_num += 1
        #Prime data
        prime_pos = self.prime.position[np.newaxis, :]
        prime_rad = self.prime.my_rad
        #COLLISIONS
        #Prime vs everyone
        if not self.prime.crashed:
            #Prime vs Invaders
            if len(all_inv_pos) > 0:
                dist_p_i = cdist(prime_pos, all_inv_pos)[0]
                if np.any(dist_p_i < prime_rad + all_inv_rad):
                    self.prime.crashed = True
            #Prime vs Pursuers
            if len(all_purs_pos) > 0 and not self.prime.crashed:
                dist_p_p = cdist(prime_pos, all_purs_pos)[0]
                if np.any(dist_p_p < prime_rad + all_purs_rad):
                    self.prime.crashed = True
            #Prime vs Obstacles
            if self.obs_centers is not None and not self.prime.crashed:
                dist_p_o = cdist(prime_pos, self.obs_centers)[0]
                if np.any(dist_p_o < prime_rad + self.obs_radii):
                    self.prime.crashed = True
        #Agents vs Obstacles
        if self.obs_centers is not None:
            #Invaders vs Obstacles
            if len(all_inv_pos) > 0:
                #matrix of distances
                dists_i_o = cdist(all_inv_pos, self.obs_centers) 
                #matrix of collision limits
                limits_i_o = all_inv_rad[:, np.newaxis] + self.obs_radii[np.newaxis, :]
                #collision of invader in any obstacle
                crashed_inv_mask = np.any(dists_i_o < limits_i_o, axis=1)
                for idx in np.where(crashed_inv_mask)[0]:
                    free_inv[idx].crashed = True
            #Pursuers vs Obstacles
            if len(all_purs_pos) > 0:
                #matrix of distances
                dists_p_o = cdist(all_purs_pos, self.obs_centers)
                #matrix of collision limits
                limits_p_o = all_purs_rad[:, np.newaxis] + self.obs_radii[np.newaxis, :]
                #collision of invader in any obstacle
                crashed_purs_mask = np.any(dists_p_o < limits_p_o, axis=1)
                for idx in np.where(crashed_purs_mask)[0]:
                    free_purs[idx].crashed = True
        #Pursuers vs Invaders
        if len(all_purs_pos) > 0 and len(all_inv_pos) > 0:
            #matrix of distances
            dists_p_i = cdist(all_purs_pos, all_inv_pos)
            #matrix of collision limits
            limits_p_i = all_purs_rad[:, np.newaxis] + all_inv_rad[np.newaxis, :]
            #capture matrix
            capture_matrix = dists_p_i < limits_p_i
            #invaders caught
            captured_inv_mask = np.any(capture_matrix, axis=0)
            #those who were caught
            for idx in np.where(captured_inv_mask)[0]:
                if not free_inv[idx].crashed:
                    self.captured_count += 1
                    free_inv[idx].crashed = True
        #Pursuers vs Pursuers
        if len(all_purs_pos) > 1:
            #matrix of distances
            dists_p_p = cdist(all_purs_pos, all_purs_pos)
            #dists of one to self set to infinity
            np.fill_diagonal(dists_p_p, np.inf)
            #matrix of collision limits
            limits_p_p = all_purs_rad[:, np.newaxis] + all_purs_rad[np.newaxis, :]
            #collision mask
            swarm_crash_mask = np.any(dists_p_p < limits_p_p, axis=1)
            #those who crashed labeled as crashed
            for idx in np.where(swarm_crash_mask)[0]:
                free_purs[idx].crashed = True
        #ending check
        done = self.prime.crashed or self.prime.finished #or (self.captured_count == self.sc.INVADER_NUM) 
        return self.get_state(), done

    def get_state(self):
        #returns state of the simulation
        return {
            "prime": self.prime.position.copy(),
            "pursuers": [p.position.copy() for p in self.pursuers],
            "pursuers_status": [[p.state, p.target] for p in self.pursuers], #for colors
            "invaders": [i.position.copy() for i in self.invaders],
            "time": self.time
        }