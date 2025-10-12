import numpy as np

class Agent:
    def __init__(self, position, speed):
        self.position = np.array(position, dtype=float)
        self.speed = speed

    def move(self, direction, dt=0.1):
        if np.linalg.norm(direction) != 0:
            direction = direction / np.linalg.norm(direction)
            self.position += self.speed * direction * dt
    #TODO
    def clip_angle(self, direction):
        return