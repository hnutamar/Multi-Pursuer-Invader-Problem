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
        # Uložíme si parametry pro reset
        self.init_params = {
            'purs_acc': purs_acc, 'prime_acc': prime_acc, 'inv_acc': inv_acc,
            'prime_pos': prime_pos, 'inv_pos': inv_pos, 'purs_pos': purs_pos, 'purs_num': purs_num
        }
        self.reset()

    def reset(self):
        """Resetuje simulaci do počátečního stavu."""
        self.time = 0.0
        self.captured_count = 0
        self.purs_crash = False
        self.prime_crashed = False
        # Inicializace agentů (převzato z tvého _init_agents)
        self._init_agents(**self.init_params)
        return self.get_state()

    def _init_agents(self, purs_acc, prime_acc, inv_acc, prime_pos, inv_pos, purs_pos, purs_num):
        # Hranice a waypoint
        x_border = self.sc.WORLD_WIDTH / 6
        y_border = self.sc.WORLD_HEIGHT / 6
        if self._3d:
            z_border = self.sc.WORLD_Z / 6
            self.way_point = np.array([self.sc.WORLD_WIDTH - x_border, self.sc.WORLD_HEIGHT - y_border, self.sc.WORLD_Z - z_border])
        else:
            # Tady předpokládám default LINE, pokud chceš měnit mody, přidej to do initu
            self.way_point = np.array([self.sc.WORLD_WIDTH - x_border, self.sc.WORLD_HEIGHT - y_border])
        # Prime unit
        acc_prime = prime_acc or 0.1
        if self._3d:
            pos_u = prime_pos if prime_pos is not None else np.array([3.0, 3.0, 3.0])
        else:
            pos_u = prime_pos if prime_pos is not None else np.array([3.0, 3.0])
            
        self.prime = Prime_unit(position=pos_u, max_acc=acc_prime, max_omega=1.0, my_rad=self.sc.UNIT_RAD)

        # Generování pozic (zkráceno pro přehlednost, zachována tvá logika)
        # ... (zde by byl tvůj kód pro rnd_points_purs a rnd_points_inv) ...
        # Pro demonstraci použijeme jednoduchý generátor, pokud nejsou zadány:
        if self._3d:
            rnd_points_purs = purs_pos if purs_pos is not None else np.random.rand(self.sc.PURSUER_NUM, 3) * 10
            rnd_points_inv = inv_pos if inv_pos is not None else np.random.rand(self.sc.INVADER_NUM, 3) * 10 + 20
        else:
            rnd_points_purs = purs_pos if purs_pos is not None else np.random.rand(self.sc.PURSUER_NUM, 2) * 10
            rnd_points_inv = inv_pos if inv_pos is not None else np.random.rand(self.sc.INVADER_NUM, 2) * 10 + 20

        rnd_acc_inv = np.full(self.sc.INVADER_NUM, inv_acc) if inv_acc is not None else np.random.uniform(1.3, 1.5, self.sc.INVADER_NUM)
        acc_purs = purs_acc or 1.0

        # Vytvoření listů agentů
        self.pursuers = []
        for i in range(self.sc.PURSUER_NUM):
            p_num = purs_num[i] if purs_num is not None else np.random.randint(0, 1001)
            p = Pursuer(position=rnd_points_purs[i], max_acc=acc_purs, max_omega=1.5, my_rad=self.sc.DRONE_RAD, purs_num=p_num)
            self.pursuers.append(p)
        self.invaders = []
        for i in range(self.sc.INVADER_NUM):
            inv = Invader(position=rnd_points_inv[i], max_acc=rnd_acc_inv[i], max_omega=1.5, my_rad=self.sc.DRONE_RAD)
            self.invaders.append(inv)
        # Překážka (pouze data, ne grafika)
        self.obstacle = None
        if not self._3d and self.sc.obs_patch is not None:
             # Uložíme si jen data potřebná pro fyziku (střed, poloměr), ne patch
             self.obstacle = type('obj', (object,), {'center': self.sc.obs_patch.center, 'radius': self.sc.obs_patch.radius})
        elif self._3d and self.sc.obs_patch is not None:
             self.obstacle = [None, self.sc.OBS_POS, self.sc.OBS_RAD]

    def step(self, dt=0.1, manual_invader_vel=None):
        """Hlavní smyčka fyziky."""
        # Filtrujeme "živé" agenty
        free_inv = [inv for inv in self.invaders if not inv.crashed]
        free_purs = [pur for pur in self.pursuers if not pur.crashed]
        # 1. Výpočet akcí (Pohyb Invaderů)
        dirs_i = []
        for idx, inv in enumerate(self.invaders):
            # Pokud je manuální ovládání pro prvního invadera
            if idx == 0 and not inv.crashed and manual_invader_vel is not None:
                acc_size = np.linalg.norm(manual_invader_vel)
                if acc_size > 0:
                    inv_acc = (manual_invader_vel / acc_size) * inv.max_acc
                else:
                    inv_acc = np.zeros_like(manual_invader_vel)
                dirs_i.append(inv_acc)
            else:
                # AI logika
                dirs_i.append(inv.evade(free_purs, self.prime, self.obstacle))
        # 2. Výpočet akcí (Pohyb Pursuerů)
        dirs_p = []
        for purs in self.pursuers:
            close_purs = [p for p in free_purs if (np.linalg.norm(p.position - purs.position) - p.my_rad - purs.my_rad) <= self.sc.PURS_VIS]
            dirs_p.append(purs.pursue(free_inv, close_purs, self.prime, self.obstacle))
        # 3. Pohyb Prime
        dir_u = self.prime.fly(self.way_point, free_inv, free_purs, Modes.LINE)
        # 4. Aplikace pohybu (Euler integration)
        for p, p_dir in zip(self.pursuers, dirs_p):
            p.move(p_dir) # Předpokládám, že move bere vektor zrychlení/rychlosti
        # Invadeři a Prime se hýbou jen po uplynutí delay (pokud to v RL chceš, jinak smaž podmínku)
        # if self.time > formation_delay... (zjednodušeno pro RL - hýbou se hned nebo podle logiky)
        for i, i_dir in zip(self.invaders, dirs_i):
            i.move(i_dir)
        self.prime.move(dir_u)

        self.time += dt

        # 5. Kolize a Logika hry (Tvá logika z update metody)
        
        # Prime vs Invader
        for i in self.invaders:
            if not i.crashed and np.sum((i.position - self.prime.position)**2) < self.sc.UNIT_DOWN_RAD**2:
                self.prime.crashed = True
            i.purs_num = 0

        # Prime vs Obstacle
        if self.obstacle is not None:
            if not self._3d and np.sum((self.prime.position - self.obstacle.center)**2) < self.obstacle.radius**2:
                self.prime.crashed = True
            elif self._3d and np.sum((self.prime.position - self.sc.OBS_POS)**2) < self.sc.OBS_RAD**2:
                self.prime.crashed = True

        #collisions check, pursue check
        for p in free_purs:
            #pursue check
            if p.target is not None:
                p.target[0].purs_num += 1
            #obstacle check
            if self.sc.obs_patch is not None:
                if not self._3d and np.sum((p.position - self.sc.obs_patch.center)**2) < self.sc.obs_patch.radius**2:
                    p.crashed = True
                elif self._3d and np.sum((p.position - self.sc.OBS_POS)**2) < self.sc.OBS_RAD**2:
                    p.crashed = True
            #capture check
            for i in free_inv:
                if np.sum((p.position - i.position)**2) < self.sc.CAPTURE_RAD**2 and not i.crashed:
                    self.captured_count += 1
                    i.crashed = True
            #crash (pursuer and pursuer)
            #if self.time > self.crash_enabled or self.purs_crash:
            #    self.purs_crash = True
            for other in free_purs:
                if (other is not p) and np.sum((p.position - other.position)**2) < self.sc.CRASH_RAD**2:
                    p.crashed = True
                    other.crashed = True
            #crash (pursuer and prime)
            if np.sum((p.position - self.prime.position)**2) < self.sc.UNIT_DOWN_RAD**2:
                self.prime.crashed = True

        # Návrat stavu pro vizualizaci nebo AI
        done = self.prime.crashed #or (self.captured_count == self.sc.INVADER_NUM)
        reward = 0 # Tady budeš později počítat RL reward
        
        return self.get_state(), reward, done

    def get_state(self):
        """Vrátí čistá data."""
        return {
            "prime": self.prime.position.copy(),
            "pursuers": [p.position.copy() for p in self.pursuers],
            "pursuers_status": [p.state for p in self.pursuers], # Pro barvičky
            "invaders": [i.position.copy() for i in self.invaders],
            "time": self.time
        }