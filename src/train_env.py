import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from world import SimulationWorld
from sim_config3D import Sim3DConfig
from stable_baselines3 import PPO
from pursuer_states import States
import torch

class FastWorldEnv(gym.Env):
    def __init__(self, world_instance: SimulationWorld, sc, test=False):
        super().__init__()
        self.world = world_instance
        self.sc = sc
        #action space - acc vector
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)
        #obs space
        #self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(101,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(102,), dtype=np.float32)
        #episode limits
        self.current_step = 0
        self.test = test
        #self.lost_purs_crash = 0
        self.lost_invader_prime = 0
        self.lost_pursuer_prime = 0
        self.lost_purs_crash = 0
        self.all_invaders_dead = 0
        if test:
            self.episode_num = 1#np.inf
            self.max_steps = 250
        else:
            self.episode_num = 1
            self.max_steps = 250
        #needed for reward
        self.last_inv_prime_dist = 60.0
        self.last_state = States.FORM
        # self.last_dist_to_inv = np.linalg.norm(self.world.pursuers[0].position - self.world.invaders[0].position)
        # self.last_inv_pos = self.world.invaders[0].position
        # self.obs_centers = []
        # self.obs_rads = []
        for i in range(self.sc.PURSUER_NUM):
            self.world.pursuers[i].is_rl_controlled = True
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #num of purs in episode
        #new_purs_num = np.random.randint(4, 21)
        new_purs_num = np.random.randint(10, 21)
        self.pursuing_purs = new_purs_num
        #self.pursuing_purs = np.random.randint(1, new_purs_num // 2) #new_purs_num // 2 
        #pursuers
        new_purs_speed = np.random.uniform(6.0, 8.0)
        new_purs_speeds = np.full(new_purs_num, new_purs_speed, dtype=np.float32)
        new_purs_acc = np.random.uniform(low=new_purs_speed / 2.0, high=new_purs_speed / 1.3)
        self.new_purs_accs = np.full(new_purs_num, new_purs_acc, dtype=np.float32)
        max_purs_speed = np.max(new_purs_speeds)
        #invaders
        new_invs_num = np.random.randint(1, min(new_purs_num, 5))
        new_inv_speed = np.random.uniform(2.0, max_purs_speed*0.9, size=new_invs_num)
        new_inv_acc = np.random.uniform(new_inv_speed / 2.0, new_inv_speed / 1.3)
        #prime
        new_prime_speed = 1.0
        new_prime_acc = np.random.uniform(new_prime_speed / 4.0, new_prime_speed / 2.0)
        #positions
        inv_pos = self.get_random_invader_start(num_invaders=new_invs_num)
        purs_positions = self.get_random_pursuer_starts(new_purs_num, inv_pos)
        #random radii
        drone_rad = random.uniform(0.1, 0.4)
        prime_rad = random.uniform(0.1, 0.7)
        #update config
        self.world.sc.PURSUER_NUM = new_purs_num
        self.world.sc.INVADER_NUM = new_invs_num
        self.world.sc.DRONE_RAD = drone_rad
        self.world.sc.UNIT_RAD = prime_rad
        #obstacles
        #num_obs = random.choice([4, 5, 6, 7, 8])
        num_obs = random.choice([0, 1, 2, 3, 4, 5])
        if num_obs > 0:
            #array of pos and radii
            all_agents_pos = np.vstack([inv_pos, purs_positions,np.array([3.0, 3.0, 7.0])])
            all_agents_rad = np.concatenate([np.full(new_purs_num + new_invs_num, drone_rad), [prime_rad]])
            #obs centers and radii
            self.obs_centers, self.obs_rads = self.generate_safe_obstacles(num_obs, all_agents_pos, all_agents_rad, 20, True)
            self.world.sc.obs_pos = self.obs_centers
            self.world.sc.obs_rads = self.obs_rads
            self.world.sc.obstacle = True
        else:
            self.world.sc.obstacle = False
            self.obs_centers = []
            self.obs_rads = []
        #brave new world
        self.world.reset(purs_acc=self.new_purs_accs, prime_acc=new_prime_acc, purs_pos=purs_positions,
            inv_acc=new_inv_acc, purs_speed=new_purs_speeds, inv_speed=new_inv_speed, prime_speed=new_prime_speed, inv_pos=inv_pos)
        for i in range(new_purs_num):
            self.world.pursuers[i].is_rl_controlled = True
        #setting invader speed from the beginning
        #inv_to_prime = self.world.prime.position - self.world.invaders[0].position
        #inv_speed = (inv_to_prime / np.linalg.norm(inv_to_prime)) * np.random.uniform(low=0.0, high=self.world.invaders[0].cruise_speed)
        #self.world.invaders[0].curr_speed = inv_speed
        #first step
        self.world.step()
        obs = self.world.pursuers[0].get_observation()
        #reseting steps
        self.current_step = 0
        self.last_inv_prime_dist = 60.0
        self.last_state = States.FORM
        # self.last_dist_to_inv = np.linalg.norm(self.world.pursuers[0].position - self.world.invaders[0].position)
        # self.last_inv_pos = self.world.invaders[0].position
        return obs, {}
    
    def load_teammate_brain(self, model_path, model_path2=None):
        rnd_num = np.random.randint(0, 2)
        #updating the brain
        if rnd_num == 0 or model_path2 is None:
            self.teammate_brain = PPO.load(model_path, device="cpu")
        else:
            self.teammate_brain = PPO.load(model_path2, device="cpu")
        #if self.episode_num > 800_000/10:
        with torch.no_grad():
            self.teammate_brain.policy.log_std.data = torch.full_like(self.teammate_brain.policy.log_std.data, -2.8)

    def generate_safe_obstacles(self, num_obs, agent_positions, agent_radii, max_coord, is_3d, min_r=1.0, max_r=5.0, safe_margin=1.5):
        #arrays
        centers = []
        radii = []
        for _ in range(num_obs):
            placed = False
            #searching only 100 times, otherwise map could be full
            for _ in range(100): 
                #random rad
                r = np.random.uniform(min_r, max_r)
                #random pos
                if is_3d:
                    c = np.random.uniform(-max_coord, max_coord, size=3)
                    c[2] = np.random.uniform(r, max_coord) 
                else:
                    c = np.random.uniform(-max_coord, max_coord, size=2)
                #collision check
                dists = np.linalg.norm(agent_positions - c, axis=1)
                #with safe margin
                safe_dists = r + agent_radii + safe_margin
                #everything safe
                if not np.any(dists < safe_dists):
                    centers.append(c)
                    radii.append(r)
                    placed = True
                    break
        return np.array(centers), np.array(radii)

    def get_random_invader_start(self, num_invaders=1):
        prime_pos = np.array([3.0, 3.0, 7.0])
        # 1. Vygenerujeme vzdálenosti pro všechny invadery najednou (sloupcový vektor)
        dists = np.random.uniform(50.0, 70.0, size=(num_invaders, 1))
        # 2. Vygenerujeme náhodné směry pro všechny naráz (matice N x 3)
        dirs = np.random.randn(num_invaders, 3)
        dirs[:, 2] = np.abs(dirs[:, 2]) # Všichni budou nahoře (Z > 0)
        # 3. Normalizace směrů (vydělíme každý řádek jeho délkou)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = dirs / norms
        # 4. Výpočet nových pozic (Prime pozice + vektor směru * vzdálenost)
        new_inv_pos = prime_pos + (dirs * dists)
        # 5. Omezení výšky (Z souřadnice nesmí klesnout pod 1.0)
        new_inv_pos[:, 2] = np.maximum(1.0, new_inv_pos[:, 2])        
        # Jinak vrátíme seznam polí (nebo můžete nechat return new_inv_pos, pokud chcete 2D Numpy matici)
        # if num_invaders == 1:
        #     return new_inv_pos[0]
        return list(new_inv_pos)

    def get_random_pursuer_starts(self, num_pursuers, inv_pos):
        # center, prime position
        prime_pos = np.array([3.0, 3.0, 7.0])    
        # array for all pursuers
        positions = np.zeros((num_pursuers, 3))
        # Vzdálenost, jak daleko od Invadera (nebo Prime) se má spawnout
        dist_close = np.random.uniform(2.2, 3.0)
        dir_close = np.random.randn(3)
        dir_close[2] = abs(dir_close[2]) # Aby se spawnoval spíš nahoře
        dir_close = dir_close / np.linalg.norm(dir_close)
        positions[0] = prime_pos + (dir_close * dist_close)
        #others can be further
        if num_pursuers > 1:
            num_far = num_pursuers - 1
            #dist from prime
            dists_far = np.random.uniform(3.0, 8.0, size=num_far)    
            #random directions
            dirs_far = np.random.randn(num_far, 3)
            dirs_far[:, 2] = np.abs(dirs_far[:, 2])   
            #normalization
            norms = np.linalg.norm(dirs_far, axis=1, keepdims=True)
            dirs_far = dirs_far / norms    
            #computing final pos
            positions[1:] = prime_pos + (dirs_far * dists_far[:, np.newaxis])    
        #safe clip
        positions[:, 2] = np.clip(positions[:, 2], 1.0, np.inf)
        #so that learning pursuer is not always by prime
        rnd_num = np.random.randint(0, 5)
        if rnd_num != 1:
            np.random.shuffle(positions)
        return positions

    def step(self, action):
        self.episode_num += 1
        self.current_step += 1
        all_invaders = self.world.free_inv
        #first pursuer is learning
        vis_inv_0 = self.world.pursuers[0].get_closest_invaders(all_invaders, 2)
        self.world.pursuers[0].set_rl_action(action, vis_inv_0)
        #other pursuers
        for i in range(1, len(self.world.pursuers)):
            if self.teammate_brain is not None:
                #AI from prev generation
                obs_i = self.world.pursuers[i].get_observation()
                #getting action
                action_i, _ = self.teammate_brain.predict(obs_i, deterministic=self.test)
                vis_inv_i = self.world.pursuers[i].get_closest_invaders(all_invaders, 2)
                self.world.pursuers[i].set_rl_action(action_i, vis_inv_i)
            else:
                #not having prev model
                self.world.pursuers[i].is_rl_controlled = False
        #frame skipping, 0.02 is too short
        min_inv_dist = np.inf
        any_invader_crashed = 0
        crashed_this_step = []
        distances = []
        for i in range(20):
            if i == 19:
                #last step
                state, done = self.world.step()
                prime_pos = state["prime"]
                for inv in self.world.free_inv:
                    dist = np.linalg.norm(inv.position - prime_pos)
                    if dist < min_inv_dist:
                        min_inv_dist = dist
                    if inv.crashed:
                        any_invader_crashed += 1
                        distances.append(dist)
                        crashed_this_step.append((inv, dist))
            else:
                #common steps
                state, done = self.world.step()
                prime_pos = state["prime"]
                for inv in self.world.free_inv:
                    if inv.crashed:
                        any_invader_crashed += 1  
                        dist = np.linalg.norm(inv.position - prime_pos)    
                        distances.append(dist) 
                        crashed_this_step.append((inv, dist))
        #computing reward
        prime_pos = state["prime"]
        target = self.world.pursuers[0].target is not None
        in_pursue = self.world.pursuers[0].state == States.PURSUE
        current_state = self.world.pursuers[0].state
        #if episode is too long
        truncated = self.current_step >= self.max_steps
        terminated = False
        reward = 0
        reward += 0.05
        #penalty for switching state too much
        if current_state != self.last_state:
            reward -= 1.0
        self.last_state = current_state
        #penalty for pursuing target with a lot of pursuers
        # if in_pursue and target:
        #     target_invader = self.world.pursuers[0].target["target"]
        #     pursuers_on_target = target_invader.purs_num
        #     if pursuers_on_target > 2:
        #         reward -= 0.5 * (pursuers_on_target - 2)
        #safe distance
        # safe_distance = min(min_inv_dist, 20.0)
        # safety_ratio = safe_distance / 20.0
        # reward += safety_ratio * 0.1 
        # if in_pursue and target:
        #     p_i_dist = np.linalg.norm(prime_pos - self.world.pursuers[0].target["target"].position)
        #     if self.world.pursuers[0].target["purs_type"] == self.world.pursuers[0].purs_types["circling"]:
        #         if p_i_dist > 18.0:
        #             reward += 0.5
        #         elif p_i_dist < 28.0:
        #             reward -= 0.5
        #     else:
        #         if p_i_dist > 30.0:
        #             reward -= 0.2
        #         else:
        #             reward += 0.5
        if min_inv_dist < 18.0:
            delta_dist = min_inv_dist - self.last_inv_prime_dist
            # Pokud se vzdálenost ZVĚTŠUJE (dron ho fyzicky odtlačil/vykolejil!)
            if delta_dist > 0:
                # Tady mu dáme pořádný násobič. Když ho nárazem odhodí o 0.5 metru, 
                # dostane okamžitý štědrý bonus. Tohle ho naučí používat kinetickou energii!
                reward += delta_dist * 5.0 
            else:
                # Vetřelec se blíží. Mírný trest, který ho nutí to zastavit.
                reward += delta_dist * 0.5  
        elif min_inv_dist >= 18.0:
            # Vetřelec je bezpečně daleko. 
            # Přesně jak jste chtěl - dáváme 1.0 za frame. Bude se snažit je tu udržet věčně.
            reward += 1.0
        #reward for being in the formation
        # ==========================================
        # ODMĚNA ZA FORMACI A JEJÍ SYMETRII
        # ==========================================
        if current_state == States.FORM:
            # 1. Plat za to, že je hlídání bezpečné (to už máte)
            # if min_inv_dist > 30.0:
            #     reward += 0.05
            # 2. BONUS ZA DOKONALOU KOULI (Triks s těžištěm)
            # Vyfiltrujeme jen ty drony, kteří aktuálně drží formaci
            form_positions = [p.position for p in self.world.free_purs if p.state == States.FORM]
            # Abychom mohli hodnotit kouli, musí v ní být rozumný počet dronů
            if len(form_positions) >= 4:
                # Spočítáme těžiště všech strážců
                centroid = np.mean(form_positions, axis=0)
                # Jak moc se těžiště roje liší od pozice šéfa?
                centroid_offset = np.linalg.norm(centroid - prime_pos)
                com_reward = max(0.0, 3.0 - centroid_offset) * 0.05
                reward += com_reward
        #reward for invader crashing
        target_dict = self.world.pursuers[0].target
        agent_pos = self.world.pursuers[0].position
        for inv, dist_to_prime in crashed_this_step:
            #my target
            is_my_target = (target_dict is not None and target_dict["target"] is inv)
            #or was I close
            dist_to_agent = np.linalg.norm(agent_pos - inv.position)
            physically_involved = dist_to_agent < 5.0
            if is_my_target or physically_involved:
                #giving reward
                if dist_to_prime > 15.0:
                    reward += 5.0
                else:
                    reward += 30.0
            else:
                #no involvement
                reward += 2.5
        if len(self.world.free_inv) == 0:
            self.all_invaders_dead += 1
            terminated = True
        #prime died
        if min_inv_dist < 1.0 or done: 
            if done:
                self.lost_pursuer_prime += 1
            else:
                self.lost_invader_prime += 1
            reward -= 100.0
            terminated = True
        #pursuer died
        if self.world.pursuers[0].crashed:
            if not done:
                self.lost_purs_crash += 1
            reward -= 35.0 
            terminated = True
        #penalty for trying attacking when too much attackers attacks
        # if self.world.pursuers[0].tried_invalid_attack:
        #     reward -= 0.05
        # Update last distance for next step
        self.last_inv_prime_dist = min(min_inv_dist, 20.0)
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        #getting observation
        obs = self.world.pursuers[0].get_observation()
        return obs

class HerdingEnv(gym.Env):
    def __init__(self, world_instance: SimulationWorld, sc, test=False):
        super().__init__()
        self.world = world_instance
        self.sc = sc
        #action space - acc vector
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        #obs space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(111,), dtype=np.float32)
        #episode limits
        self.current_step = 0
        self.test = test
        #self.lost_purs_crash = 0
        self.lost_invader_prime = 0
        self.lost_pursuer_prime = 0
        self.lost_purs_crash = 0
        if test:
            self.episode_num = 1#np.inf
            self.max_steps = 500
        else:
            self.episode_num = 1
            self.max_steps = 1000
        #needed for reward
        self.last_inv_prime_dist = np.linalg.norm(self.world.prime.position - self.world.invaders[0].position)
        self.last_dist_to_inv = np.linalg.norm(self.world.pursuers[0].position - self.world.invaders[0].position)
        self.last_inv_pos = self.world.invaders[0].position
        self.obs_centers = []
        self.obs_rads = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #num of purs in episode
        #new_purs_num = np.random.randint(4, 21)
        new_purs_num = np.random.randint(1, 5)
        self.pursuing_purs = new_purs_num
        #self.pursuing_purs = np.random.randint(1, new_purs_num // 2) #new_purs_num // 2 
        #pursuers
        new_purs_speed = np.random.uniform(6.0, 8.0)
        new_purs_speeds = np.full(new_purs_num, new_purs_speed, dtype=np.float32)
        new_purs_acc = np.random.uniform(low=new_purs_speed / 2.0, high=new_purs_speed / 1.3)
        self.new_purs_accs = np.full(new_purs_num, new_purs_acc, dtype=np.float32)
        max_purs_speed = np.max(new_purs_speeds)
        #invaders
        new_inv_speed = np.random.uniform(3.0, max_purs_speed*0.9)
        new_inv_acc = np.random.uniform(new_inv_speed / 2.0, new_inv_speed / 1.3)
        #prime
        new_prime_speed = 1.0
        new_prime_acc = np.random.uniform(new_prime_speed / 4.0, new_prime_speed / 2.0)
        #positions
        inv_pos = self.get_random_invader_start()
        purs_positions = self.get_random_pursuer_starts(new_purs_num, inv_pos)
        #random radii
        drone_rad = random.uniform(0.1, 0.4)
        prime_rad = random.uniform(0.1, 0.7)
        #update config
        self.world.sc.PURSUER_NUM = new_purs_num
        self.world.sc.DRONE_RAD = drone_rad
        self.world.sc.UNIT_RAD = prime_rad
        #obstacles
        num_obs = random.choice([4, 5, 6, 7, 8])
        #num_obs = random.choice([1, 2, 3, 4, 5])
        if num_obs > 0:
            #array of pos and radii
            all_agents_pos = np.vstack([inv_pos, purs_positions,np.array([3.0, 3.0, 7.0])])
            all_agents_rad = np.concatenate([[drone_rad],np.full(new_purs_num, drone_rad), [prime_rad]])
            #obs centers and radii
            self.obs_centers, self.obs_rads = self.generate_safe_obstacles(num_obs, all_agents_pos, all_agents_rad, 20, True)
            self.world.sc.obs_pos = self.obs_centers
            self.world.sc.obs_rads = self.obs_rads
            self.world.sc.obstacle = True
        else:
            self.world.sc.obstacle = False
            self.obs_centers = []
            self.obs_rads = []
        #brave new world
        self.world.reset(purs_acc=self.new_purs_accs, prime_acc=new_prime_acc, purs_pos=purs_positions,
            inv_acc=[new_inv_acc], purs_speed=new_purs_speeds, inv_speed=[new_inv_speed], prime_speed=new_prime_speed, inv_pos=[inv_pos])
        #setting invader speed from the beginning
        inv_to_prime = self.world.prime.position - self.world.invaders[0].position
        inv_speed = (inv_to_prime / np.linalg.norm(inv_to_prime)) * np.random.uniform(low=0.0, high=self.world.invaders[0].cruise_speed)
        self.world.invaders[0].curr_speed = inv_speed
        #setting to pursue state
        for i in range(self.pursuing_purs):
            self.world.pursuers[i].state = States.PURSUE
            self.world.pursuers[i].target = {"target": self.world.invaders[0], "tar_pos": self.world.invaders[0].position, "tar_vel": self.world.invaders[0].curr_speed,
                                             "tar_rad": self.world.invaders[0].my_rad, "tar_ang": self.world.invaders[0].curr_omega, "tar_acc": self.world.invaders[0].curr_acc, "purs_type": 1}
        #first step
        self.world.step()
        obs = self.world.pursuers[0].get_observation_herding()
        #reseting steps
        self.current_step = 0
        self.last_inv_prime_dist = np.linalg.norm(self.world.prime.position - self.world.invaders[0].position)
        self.last_dist_to_inv = np.linalg.norm(self.world.pursuers[0].position - self.world.invaders[0].position)
        self.last_inv_pos = self.world.invaders[0].position
        return obs, {}
    
    def load_teammate_brain(self, model_path, model_path2=None):
        rnd_num = np.random.randint(0, 2)
        #updating the brain
        if rnd_num == 0 or model_path2 is None:
            self.teammate_brain = PPO.load(model_path, device="cpu")
        else:
            self.teammate_brain = PPO.load(model_path2, device="cpu")
        #if self.episode_num > 800_000/10:
        with torch.no_grad():
            self.teammate_brain.policy.log_std.data = torch.full_like(self.teammate_brain.policy.log_std.data, -2.8)

    def generate_safe_obstacles(self, num_obs, agent_positions, agent_radii, max_coord, is_3d, min_r=1.0, max_r=5.0, safe_margin=1.5):
        #arrays
        centers = []
        radii = []
        for _ in range(num_obs):
            placed = False
            #searching only 100 times, otherwise map could be full
            for _ in range(100): 
                #random rad
                r = np.random.uniform(min_r, max_r)
                #random pos
                if is_3d:
                    c = np.random.uniform(-max_coord, max_coord, size=3)
                    c[2] = np.random.uniform(r, max_coord) 
                else:
                    c = np.random.uniform(-max_coord, max_coord, size=2)
                #collision check
                dists = np.linalg.norm(agent_positions - c, axis=1)
                #with safe margin
                safe_dists = r + agent_radii + safe_margin
                #everything safe
                if not np.any(dists < safe_dists):
                    centers.append(c)
                    radii.append(r)
                    placed = True
                    break
        return np.array(centers), np.array(radii)

    def get_random_invader_start(self):
        #random pos of invader, not too far or too close
        prime_pos = np.array([3.0, 3.0, 7.0])
        dist = np.random.uniform(17.0, 25.0)
        dir = np.random.randn(3)
        dir[2] = abs(dir[2]) 
        dir = dir / np.linalg.norm(dir)
        new_inv_pos = prime_pos + (dir * dist)
        new_inv_pos[2] = max(1.0, new_inv_pos[2])
        return new_inv_pos

    def get_random_pursuer_starts(self, num_pursuers, inv_pos):
        # center, prime position
        prime_pos = np.array([3.0, 3.0, 7.0])    
        # array for all pursuers
        positions = np.zeros((num_pursuers, 3))
        # Vzdálenost, jak daleko od Invadera (nebo Prime) se má spawnout
        dist_close = np.random.uniform(2.2, 3.0)
        # if self.episode_num <= 500_000 / 10 and not self.test:
        #     # --- FÁZE 1: SPAWN PŘESNĚ NA LOS (MEZI INVADERA A PRIME) ---
        #     # 1. Směrový vektor: Od Invadera směrem k Prime dronovi
        #     dir_to_prime = prime_pos - inv_pos
        #     # 2. Normalizace: Uděláme z vektoru šipku o délce přesně 1
        #     dist_to_prime = np.linalg.norm(dir_to_prime)
        #     if dist_to_prime > 0:
        #         norm_dir = dir_to_prime / dist_to_prime
        #     else:
        #         norm_dir = np.array([0.0, 0.0, 1.0]) # Fallback    
        #     # 3. Naspawnujeme Pursuera na Invadera a posuneme ho po LOS blíž k Prime
        #     positions[0] = inv_pos + (norm_dir * dist_close)
        # else:
        #--- FÁZE 2: NÁHODNÝ SPAWN U PRIME DRONA ---
        dir_close = np.random.randn(3)
        dir_close[2] = abs(dir_close[2]) # Aby se spawnoval spíš nahoře
        dir_close = dir_close / np.linalg.norm(dir_close)
        positions[0] = prime_pos + (dir_close * dist_close)
        #others can be further
        if num_pursuers > 1:
            num_far = num_pursuers - 1
            #dist from prime
            dists_far = np.random.uniform(3.0, 6.0, size=num_far)    
            #random directions
            dirs_far = np.random.randn(num_far, 3)
            dirs_far[:, 2] = np.abs(dirs_far[:, 2])   
            #normalization
            norms = np.linalg.norm(dirs_far, axis=1, keepdims=True)
            dirs_far = dirs_far / norms    
            #computing final pos
            positions[1:] = prime_pos + (dirs_far * dists_far[:, np.newaxis])    
        #safe clip
        positions[:, 2] = np.clip(positions[:, 2], 1.0, np.inf)
        #so that learning pursuer is not always by prime
        rnd_num = np.random.randint(0, 2)
        if rnd_num != 1:
            np.random.shuffle(positions)
        return positions

    def step(self, action):
        self.episode_num += 1
        self.current_step += 1
        all_raw_actions = [action]
        #generating action of others
        if hasattr(self, 'teammate_brain') and self.teammate_brain is not None:
            for i in range(1, self.pursuing_purs):
                obs_i = self.world.pursuers[i].get_observation_herding() 
                action_i, _ = self.teammate_brain.predict(obs_i, deterministic=self.test)
                all_raw_actions.append(action_i)
        else:
            #does nothing, if there is no brain
            for i in range(1, len(self.world.pursuers)):
                all_raw_actions.append(np.zeros_like(action))
        #drone moving
        for i in range(0, self.pursuing_purs):
            raw_act = np.array(all_raw_actions[i], dtype=np.float32)
            #cliping
            norm = np.linalg.norm(raw_act)
            if norm > 1.0:
                raw_act = raw_act / norm
            target_acc = raw_act * self.new_purs_accs[i]
            #herding
            self.world.pursuers[i].herding(target_acc)
        #frame skipping, 0.02 is too short
        self.world.step()    
        self.world.step()    
        self.world.step()    
        self.world.step()      
        state, done = self.world.step()
        #computing reward
        prime_pos = state["prime"]
        invader_pos = state["invaders"][0]
        invader_crashed = state["invaders_status"][0]
        pursuer_pos = state["pursuers"][0]
        pursuer_rad = self.world.pursuers[0].my_rad
        #dist between prime nad invader
        current_inv_prime_dist = np.linalg.norm(invader_pos - prime_pos)
        #if episode is too long
        truncated = self.current_step >= self.max_steps
        reward = 0.0
        reward += 0.1
        terminated = False
        curr_dist_to_inv = np.linalg.norm(pursuer_pos - invader_pos) - 8.0
        #navigating penalty to invader
        if curr_dist_to_inv > 0:
            distance_penalty = curr_dist_to_inv * 0.05
            reward -= distance_penalty
        pursuer_positions = np.array([p.position for p in self.world.free_purs])
        #pursuer penalty
        colleague_penalty = 0.0
        safe_drone_dist = 2.0
        other_rads = np.array([p.my_rad for p in self.world.free_purs[1:]])
        other_pos = pursuer_positions[1:]
        if len(other_pos) > 0:
            distances = np.linalg.norm(other_pos - pursuer_pos, axis=1) - pursuer_rad - other_rads
            violations = safe_drone_dist - distances
            colleague_penalty = -np.sum(violations[violations > 0]) * 0.1
            reward += colleague_penalty
        #obstacle penalty
        safe_drone_dist = 4.0
        if len(self.obs_centers) > 0:
            obs_centers_arr = self.obs_centers
            obs_rads_arr = self.obs_rads
            obs_distances = np.linalg.norm(obs_centers_arr - pursuer_pos, axis=1) - pursuer_rad - obs_rads_arr
            obs_violations = safe_drone_dist - obs_distances
            obs_penalty = -np.sum(obs_violations[obs_violations > 0]) * 0.1
            reward += obs_penalty
        #ground penalty
        safe_drone_dist = 2.0
        ground_dist = pursuer_pos[2] - pursuer_rad
        if ground_dist < safe_drone_dist:
            ground_penalty = (safe_drone_dist - ground_dist) * 0.1
            reward -= ground_penalty
        #prime penalty
        safe_drone_dist = 1.0
        dist_to_prime = np.linalg.norm(pursuer_pos - self.world.prime.position) - pursuer_rad - self.world.prime.my_rad
        if dist_to_prime < safe_drone_dist:
            prime_violation = safe_drone_dist - dist_to_prime
            reward -= prime_violation * 0.2      
        #COM reward
        # if len(pursuer_positions) > 1:
        #     center_of_mass = np.mean(pursuer_positions, axis=0)
        #     invader_com_dist = np.linalg.norm(center_of_mass - invader_pos)
        #     com_reward = max(0.0, 5.0 - invader_com_dist) * 0.1
        #     reward += com_reward
        #reward for pushing invader away
        # critical_zone = 12.0
        # if current_inv_prime_dist < critical_zone:
        #     panic_penalty = ((critical_zone - current_inv_prime_dist) / critical_zone) * 0.03
        #     reward -= panic_penalty
        #reward, positive if invader is further away from prime
        diff = current_inv_prime_dist - self.last_inv_prime_dist
        if current_inv_prime_dist < 20.0: # and diff > 0:
            reward += diff * 0.1
        #if invader crashed, it is good, but better to push him away
        #if np.linalg.norm(pursuer_pos - invader_pos) - 2*pursuer_rad < 0.75:
        if invader_crashed:
            reward += 10.0
            terminated = True
        #penalization for crash
        if self.world.pursuers[0].crashed:
            #print("lost")
            if not done:
                self.lost_purs_crash += 1
            reward -= 30.0
            terminated = True
        #penalization for breaking the defense
        if current_inv_prime_dist < 1.0 or done: 
            if done:
                self.lost_pursuer_prime += 1
            else:
                self.lost_invader_prime += 1
            #print("lost")
            #if not self.world.invaders[0].crashed:
            #    self.lost += 1
            reward -= 40.0
            terminated = True
        #reward for getting invader far
        # safe_distance = min(current_inv_prime_dist, 20.0)
        # safety_ratio = safe_distance / 20.0
        # reward += safety_ratio * 0.05
        # action_penalty = np.sum(np.square(action)) * 0.005
        # reward -= action_penalty
        #penalty for invader moving too much
        # if current_inv_prime_dist > 20:
        #     inv_diff = np.linalg.norm(self.last_inv_pos - invader_pos)
        #     reward -= inv_diff * 0.05
        #self.last_inv_pos = invader_pos
        if current_inv_prime_dist > 20.0:
            reward += 0.1
        #whole game won
        #if truncated:
            #reward += min(current_inv_prime_dist, 20.0) * 2
        self.last_inv_prime_dist = current_inv_prime_dist    
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        #getting observation
        obs = self.world.pursuers[0].get_observation_herding()
        return obs
    
#UNUSED CODE:
        #phase one, invader is basically on a spot
        # if self.episode_num <= 600000:
        #     new_inv_speed = random.uniform(1.0, 2.0)
        #     new_prime_speed = 0.05
        # #phase two, invader is slowly moving
        # elif 600000 < self.episode_num <= 1300000:
        #     new_inv_speed = random.uniform(2.0, (new_purs_speed / 1300000) * self.episode_num)
        #     new_prime_speed = max((1.0 / 1300000) * self.episode_num, 0.05)
        #phase three, small change in speed
        # elif 800000 < self.episode_num <= 1300000:
        #     progress = (self.episode_num - 800000) / 500000.0    
        #     max_allowed_speed = 2.0 + progress * (new_purs_speed - 1.0 - 1.0)    
        #     new_inv_speed = random.uniform(2.0, max_allowed_speed)
        #     new_prime_speed = 1.0
        #phase four, hardcore
        #else:
        
        #gass leak
        #action_penalty = np.sum(np.square(action)) * 0.01 
        #reward -= action_penalty
        #time penalization
        #reward -= 0.1 
        #curr_dist_to_inv = np.linalg.norm(pursuer_pos - invader_pos)
        #navigating penalty to invader
        #distance_penalty = curr_dist_to_inv * 0.01
        #reward -= distance_penalty
        #penalty for pushing invader away (zero from certain distance)
        # SAFE_RADIUS = 20.0
        # invader_threat_level = max(0.0, SAFE_RADIUS - current_inv_prime_dist)
        # reward -= invader_threat_level * 0.2
        #updating for next step
        #self.last_dist_to_inv = curr_dist_to_inv