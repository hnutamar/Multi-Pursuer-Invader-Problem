
class Sim2DConfig:
    def __init__(self, dt=0.1, purs_num=15, inv_num=5, world_width=30, world_height=30, obstacle=False, obstacle_rad=0.6, obstacle_pos=[15.0, 15.0]):
        #world size
        self.WORLD_WIDTH = world_width
        self.WORLD_HEIGHT = world_height
        #time
        self.DT = dt
        #obstacles
        self.obstacle = obstacle
        self.obs_pos = obstacle_pos
        self.obs_rads = obstacle_rad
        #drone radii
        self.DRONE_RAD = 0.3
        self.UNIT_RAD = 0.6
        #drone number
        self.PURSUER_NUM = purs_num
        self.INVADER_NUM = inv_num
        #visibility of pursuer of other pursuers
        self.PURS_VIS = 2.0
        #vortex fields size
        self.FIELD_RES = 15
        self.FIELD_SIZE = 4
        self.INV_FIELD_RES = 8
        self.INV_FIELD_SIZE = 2
