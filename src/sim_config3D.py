
class Sim3DConfig:
    def __init__(self, dt=0.1, purs_num=15, inv_num=5, world_width=30, world_height=30, world_z=10, obstacle=False, obstacle_pos=None, obstacle_rad=0.6):
        #world size
        self.WORLD_WIDTH = world_width
        self.WORLD_HEIGHT = world_height
        self.WORLD_Z = world_z
        #time
        self.DT = dt
        #obstacles
        self.obstacle = obstacle
        self.obs_pos = obstacle_pos
        self.obs_rads = obstacle_rad
        #drone radii
        self.DRONE_RAD = 0.2
        self.UNIT_RAD = 0.3
        #drone number
        self.PURSUER_NUM = purs_num
        self.INVADER_NUM = inv_num
        #visibility of pursuer of other pursuers
        self.PURS_VIS = 2.0
        #marker for visualization
        self.drone_marker_size = self.DRONE_RAD * 10
        self.prime_marker_size = self.UNIT_RAD * 10

