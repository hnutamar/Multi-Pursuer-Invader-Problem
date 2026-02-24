import numpy as np
from collections import deque

class Agent:
    def __init__(self, position, max_speed, max_acc, max_omega, dt, my_rad, num_iter=0):
        #drone radius
        self.my_rad = my_rad
        #drone position
        self.position = np.array(position, dtype=float)
        self.prev_pos = np.array(position, dtype=float)
        self.path_history = deque(maxlen=20)
        self.pos_length = len(self.position)
        #drone speed and velocity
        self.max_speed = float(max_speed)
        self.max_acc = float(max_acc)
        self.curr_speed = np.zeros_like(self.position)
        self.curr_acc = np.zeros_like(self.position)
        self.cruise_speed = 0.75 * self.max_speed
        #drone controller
        self.CD = self.max_acc / (self.max_speed ** 2)
        #angular speed
        self.max_omega = max_omega
        #internal clock
        self.dt = dt
        self.num_iter = num_iter
        self.low_clock = self.dt - 0.2*self.dt
        self.high_clock = self.dt + 0.2*self.dt
        #boolean signalling crash
        self.crashed = False
        #only for prime
        self.finished = False

    def move(self, acc):
        #clock
        self.num_iter += np.random.uniform(low=self.low_clock, high=self.high_clock)
        if self.crashed:
            return
        if self.finished:
            self.curr_speed[:] = 0.0
            self.curr_acc[:] = 0.0
            return
        #safety
        if acc is None:
            acc = np.zeros_like(self.curr_speed)
        #acc cant be bigger than max acc
        acc_norm = np.linalg.norm(acc)
        if acc_norm > self.max_acc:
           acc = (acc / acc_norm) * self.max_acc
        #drag
        speed_val = np.linalg.norm(self.curr_speed)
        drag = self.CD * speed_val * self.curr_speed
        #final acc
        v_dot = acc - drag 
        #computing the final velocity
        final_v = self.curr_speed + v_dot * self.dt
        #clipping angle
        if speed_val > 0.001: 
             final_v = self.clip_angle(final_v, self.dt)
        #changing the states
        self.curr_acc = (final_v - self.curr_speed) / self.dt
        self.prev_pos = self.position
        self.position += final_v * self.dt
        self.curr_speed = final_v
        self.path_history.append(self.position.copy())
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
    
    def clip_angle(self, dir_vec, dt):
        #speed new and old
        current_speed_norm = np.linalg.norm(self.curr_speed)
        target_speed_norm = np.linalg.norm(dir_vec)
        if current_speed_norm < 1e-6 or target_speed_norm < 1e-6:
            return dir_vec
        #norming the velocity vectors, calculating angles
        u = self.curr_speed / current_speed_norm
        v = dir_vec / target_speed_norm
        dot_product = np.dot(u, v)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)
        max_step = self.max_omega * dt
        #angle is ok
        if angle <= max_step:
            return dir_vec
        #too sharp angle
        else:
            t = max_step / angle
            sin_angle = np.sin(angle)
            w1 = np.sin((1 - t) * angle) / sin_angle
            w2 = np.sin(t * angle) / sin_angle
            new_dir_normed = w1 * u + w2 * v
            return new_dir_normed * target_speed_norm