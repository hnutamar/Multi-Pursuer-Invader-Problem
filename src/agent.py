import numpy as np

class Agent:
    def __init__(self, position, max_acc, max_omega):
        self.position = np.array(position, dtype=float)
        self.max_acc = max_acc
        self.curr_speed = np.zeros_like(self.position)
        self.curr_acc = np.zeros_like(self.position)
        self.drone_dir = np.zeros_like(self.position)
        self.max_omega = max_omega
        self.CD = 0.3
        self.max_speed = self.max_acc / self.CD
        self.dt = 0.1

    def move(self, acc):
        dirs = acc
        #double integrator
        if np.linalg.norm(acc) == 0:
            return
        if np.linalg.norm(acc) > self.max_acc:
            acc = (acc / np.linalg.norm(acc)) * self.max_acc
        v = acc - self.CD * np.linalg.norm(self.curr_speed) * self.curr_speed
        final_v = self.curr_speed + v * self.dt
        final_v = self.clip_angle(final_v, self.dt)
        self.curr_acc = (final_v - self.curr_speed) / self.dt
        self.position += final_v * self.dt
        self.curr_speed = final_v
        #single integrator
        #if dist != 0:
        #    if dist < 1:
        #        final_dir = direction * dt
        #        final_dir = self.clip_angle(final_dir, dt)
        #        self.position += final_dir
        #    else:
        #        direction = direction / dist
        #        final_dir = self.max_speed * direction * dt
        #        final_dir = self.clip_angle(final_dir, dt)
        #        self.position += final_dir

    # def clip_angle(self, dir, dt):
    #     theta = np.atan2(self.curr_speed[1], self.curr_speed[0])
    #     theta_des = np.atan2(dir[1], dir[0])
    #     delta_theta = np.arctan2(np.sin(theta_des - theta), np.cos(theta_des - theta))
    #     max_omega = self.max_omega * dt
    #     delta_theta = np.clip(delta_theta, -max_omega, max_omega)
    #     theta_next = theta + delta_theta
    #     v_next = np.linalg.norm(dir) * np.array([np.cos(theta_next), np.sin(theta_next)])
    #     #self.curr_speed = v_next
    #     return v_next
    
    def clip_angle(self, dir_vec, dt):
        """
        Omezí změnu směru vektoru rychlosti podle maximální úhlové rychlosti.
        Funguje ve 2D i 3D.
        """
        # 1. Zjistíme velikosti vektorů
        current_speed_norm = np.linalg.norm(self.curr_speed)
        target_speed_norm = np.linalg.norm(dir_vec)

        # Ochrana: Pokud stojíme nebo je cíl nulový, není co točit
        if current_speed_norm < 1e-6 or target_speed_norm < 1e-6:
            return dir_vec

        # 2. Normalizujeme vektory (získáme jen směr o délce 1)
        u = self.curr_speed / current_speed_norm
        v = dir_vec / target_speed_norm

        # 3. Spočítáme úhel mezi aktuálním směrem (u) a cílem (v)
        # Používáme skalární součin: a . b = |a|*|b|*cos(alpha)
        dot_product = np.dot(u, v)
        
        # Ochrana pro numerickou stabilitu (aby arccos nehodil chybu pro 1.0000001)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        angle = np.arccos(dot_product) # Celkový úhel, o který se musíme otočit

        # 4. Maximální povolené otočení v tomto kroku
        max_step = self.max_omega * dt

        # 5. Rozhodování
        if angle <= max_step:
            # Stihneme se otočit úplně -> vracíme cílový směr
            # (zachováváme velikost podle vstupu dir_vec, stejně jako v původním kódu)
            return dir_vec
        
        else:
            # Nestihneme to -> musíme interpolovat (SLERP - Spherical Linear Interpolation)
            # Poměr t říká, jakou část cesty (angle) urazíme (max_step)
            t = max_step / angle
            
            # Vzorec pro SLERP:
            sin_angle = np.sin(angle)
            # Váhy pro starý směr a nový směr
            w1 = np.sin((1 - t) * angle) / sin_angle
            w2 = np.sin(t * angle) / sin_angle
            
            # Výsledný směrový vektor (kombinace obou)
            new_dir_normed = w1 * u + w2 * v
            
            # Vrátíme vektor se směrem new_dir, ale velikostí podle původního požadavku dir_vec
            return new_dir_normed * target_speed_norm