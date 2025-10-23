import numpy as np

class Agent:
    def __init__(self, position, max_acc, max_omega):
        self.position = np.array(position, dtype=float)
        self.max_acc = max_acc
        self.curr_speed = np.random.uniform(0.1, 0.3, size=2)
        self.curr_acc = np.array([0.0, 0.0])
        self.drone_dir = np.random.uniform(-1, 1, size=2)
        self.max_omega = max_omega
        self.CD = 0.3
        self.max_speed = self.max_acc / self.CD

    def move(self, acc, dt=0.1):
        dirs = acc
        #double integrator
        if np.linalg.norm(acc) == 0:
            return
        if np.linalg.norm(acc) > self.max_acc:
            acc = (acc / np.linalg.norm(acc)) * self.max_acc
        v = acc - self.CD * np.linalg.norm(self.curr_speed) * self.curr_speed
        final_v = self.curr_speed + v * dt
        final_v = self.clip_angle(final_v, dt)
        self.curr_acc = (final_v - self.curr_speed) / dt
        self.position += final_v * dt
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

    def clip_angle(self, dir, dt):
        theta = np.atan2(self.curr_speed[1], self.curr_speed[0])
        theta_des = np.atan2(dir[1], dir[0])
        delta_theta = np.arctan2(np.sin(theta_des - theta), np.cos(theta_des - theta))
        max_omega = self.max_omega * dt
        delta_theta = np.clip(delta_theta, -max_omega, max_omega)
        theta_next = theta + delta_theta
        v_next = np.linalg.norm(dir) * np.array([np.cos(theta_next), np.sin(theta_next)])
        #self.curr_speed = v_next
        return v_next