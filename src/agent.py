import numpy as np

class Agent:
    def __init__(self, position, speed, max_omega):
        self.position = np.array(position, dtype=float)
        self.speed = speed
        self.drone_dir = np.random.uniform(-1, 1, size=2)
        self.max_omega = max_omega

    def move(self, direction, dt=0.1):
        if np.linalg.norm(direction) != 0:
            direction = direction / np.linalg.norm(direction)
            final_dir = self.clip_angle(self.speed * direction * dt, dt)
            self.position += final_dir

    def clip_angle(self, dir, dt):
        theta = np.atan2(self.drone_dir[1], self.drone_dir[0])
        theta_des = np.atan2(dir[1], dir[0])
        delta_theta = np.arctan2(np.sin(theta_des - theta), np.cos(theta_des - theta))
        max_omega = self.max_omega * dt
        delta_theta = np.clip(delta_theta, -max_omega, max_omega)
        theta_next = theta + delta_theta
        v_next = np.linalg.norm(dir) * np.array([np.cos(theta_next), np.sin(theta_next)])
        self.drone_dir = v_next
        return v_next