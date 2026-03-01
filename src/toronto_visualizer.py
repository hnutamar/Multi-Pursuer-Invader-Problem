import numpy as np
import pybullet as p
from pursuer_states import States
import math
import random

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics

class TorontoVisualizer:
    def __init__(self, sc_config, init_state, _3d=False):
        self.sc = sc_config
        self._3d = _3d
        self.is_open = True
        #drone numbers
        self.num_pursuers = self.sc.PURSUER_NUM
        self.num_invaders = self.sc.INVADER_NUM
        self.total_drones = self.num_pursuers + self.num_invaders + 1
        #for manual control
        self.manual_vel = np.zeros(3)
        #for dynamic camera
        self.camera_yaw = 0.0
        #colors
        self.C_RED = [0.84, 0.15, 0.15, 1.0]
        self.C_BLUE = [0.12, 0.46, 0.7, 1.0]
        self.C_GREEN = [0.06, 0.92, 0.13, 1.0]
        self.C_YELLOW = [1.0, 0.93, 0.0, 1.0]
        self.C_BLACK = [0.0, 0.0, 0.0, 1.0]
        self.C_GREY = [0.5, 0.5, 0.5, 0.8]
        self.current_colors = [None] * self.total_drones
        #starting positions
        target_positions = []
        target_positions.extend(init_state['pursuers'])
        target_positions.extend(init_state['invaders'])
        target_positions.append(init_state['prime'])
        #init the animation with these positions
        INIT_XYZS = np.zeros((self.total_drones, 3))
        for i, pos in enumerate(target_positions):
            INIT_XYZS[i, 0] = pos[0]
            INIT_XYZS[i, 1] = pos[1]
            INIT_XYZS[i, 2] = pos[2] if self._3d else 2.0
        #rotation
        INIT_RPYS = np.zeros((self.total_drones, 3))
        #create aviary
        self.env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=self.total_drones, initial_xyzs=INIT_XYZS,
                              initial_rpys=INIT_RPYS, physics=Physics.PYB, gui=True, record=False)
        #pid controller
        self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for _ in range(self.total_drones)]
        #side pannels closed
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.env.CLIENT)
        #camera settings
        p.resetDebugVisualizerCamera(cameraDistance=25, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[self.sc.WORLD_WIDTH/2, 
                                    self.sc.WORLD_HEIGHT/2, 2], physicsClientId=self.env.CLIENT)
        #action matrix
        self.action = np.zeros((self.total_drones, 4))
        self.obs, _, _, _, _ = self.env.step(self.action)
        #create obstacles
        if self.sc.obstacle:
            #every obstacle
            for i in range(len(self.sc.obs_rads)):
                pos = self.sc.obs_pos[i]
                rad = self.sc.obs_rads[i]
                #3D or 2D
                pos_3d = list(pos) if self._3d else [pos[0], pos[1], 2.0]
                #circle
                visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=rad, rgbaColor=self.C_GREY, 
                    physicsClientId=self.env.CLIENT)
                #no gravity
                p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_id, basePosition=pos_3d, 
                    physicsClientId=self.env.CLIENT)
        #random wind parameters
        self.wind_active = False
        self.wind_timer = 0
        self.current_wind_force = [0.0, 0.0, 0.0]

    def render(self, state, world_instance=None):
        if not self.is_open:
            return False, None
        #ESC pressed, animation over
        keys = p.getKeyboardEvents(physicsClientId=self.env.CLIENT)
        if 27 in keys and (keys[27] & p.KEY_IS_DOWN):
            self.is_open = False
            return False, None
        #DYNAMIC CAMERA
        prime_idx = self.total_drones - 1
        prime_pos = self.env.pos[prime_idx].copy()
        #prime_idx = 0
        #prime_pos = self.env.pos[0].copy()
        #prime_idx = self.num_pursuers
        #prime_pos = self.env.pos[prime_idx].copy()
        #pos of prime
        vx = self.env.vel[prime_idx][0]
        vy = self.env.vel[prime_idx][1]
        speed = math.hypot(vx, vy)
        #only when drone moves forward
        if speed > 0.5:
            target_yaw = math.degrees(math.atan2(vy, vx)) - 90
            yaw_diff = (target_yaw - self.camera_yaw + 180) % 360 - 180
            self.camera_yaw += yaw_diff * 0.1
        #setting the camera
        # p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=self.camera_yaw, cameraPitch=-15,
        #     cameraTargetPosition=[prime_pos[0], prime_pos[1], prime_pos[2] if self._3d else 2.0], physicsClientId=self.env.CLIENT)
        #lists of drone positions
        target_positions = []
        target_positions.extend(state['pursuers'])
        target_positions.extend(state['invaders'])
        target_positions.append(state['prime'])
        #Toronto requires update much more frequent, doing the computation
        substeps = int(self.sc.DT / self.env.CTRL_TIMESTEP)
        if substeps < 1: substeps = 1
        #doing the update times the substep
        for _ in range(substeps):
            for i in range(self.total_drones):
                #having the target position, computing the control
                pos = target_positions[i]
                target_pos_3d = np.array([pos[0], pos[1], pos[2] if self._3d else 2.0])
                #adding noise
                noisy_obs = self.obs[i].copy()
                noisy_obs[0:3] += np.random.normal(0, 0.05, size=3)   # Position: 5 cm
                noisy_obs[10:13] += np.random.normal(0, 0.1, size=3) # Velocity: 10 cm/s
                self.action[i, :], _, _ = self.ctrl[i].computeControlFromState(control_timestep=self.env.CTRL_TIMESTEP,
                                          state=noisy_obs,target_pos=target_pos_3d)
            # #dynamic wind
            if not self.wind_active and random.random() < 0.002:
                self.wind_active = True
                #length of the wind
                self.wind_timer = random.randint(20, 60) 
                #force of the wind
                fx = random.uniform(-0.1, 0.1)
                fy = random.uniform(-0.1, 0.1)
                self.current_wind_force = [fx, fy, 0.0]
                print(f"Wind: X={fx:.3f} N, Y={fy:.3f} N")
            #if wind is active, apply on all drones
            if self.wind_active:
                for i in range(self.total_drones):
                    p.applyExternalForce(objectUniqueId=self.env.DRONE_IDS[i], linkIndex=-1, 
                        forceObj=self.current_wind_force, posObj=[0, 0, 0], flags=p.LINK_FRAME, physicsClientId=self.env.CLIENT)
                #subtracting the timer
                self.wind_timer -= 1
                if self.wind_timer <= 0:
                    #wind is over
                    self.wind_active = False
            #doing the physics step
            self.obs, _, _, _, _ = self.env.step(self.action)
        #removing axes at drones
        p.removeAllUserDebugItems(physicsClientId=self.env.CLIENT)
        #coloring of pursuers
        for i, p_state in enumerate(state['pursuers_status']):
            color = self.C_RED
            if p_state[0] == States.PURSUE:
                color = self.C_BLACK if p_state[1][1] == 1 else self.C_YELLOW
            self._set_color(i, color)
        #coloring of invaders
        for i, crashed in enumerate(state['invaders_status']):
            if crashed:
                self._set_color(self.num_pursuers + i, self.C_GREY)
            else:
                self._set_color(self.num_pursuers + i, self.C_BLUE)
        #coloring of prime
        self._set_color(self.total_drones - 1, self.C_GREEN)
        #real state for sync
        real_state = {'pursuers': [], 'invaders': [], 'prime': None}
        POS_NOISE_STD = 0.5  # Position: 50 cm
        VEL_NOISE_STD = 0.5  # Velocity: 50 cm/s
        #local function for computing noisy data
        def get_noisy_state(drone_idx):
            noisy_pos = self.env.pos[drone_idx].copy() + np.random.normal(0, POS_NOISE_STD, 3)
            noisy_vel = self.env.vel[drone_idx].copy() + np.random.normal(0, VEL_NOISE_STD, 3)
            return {'pos': noisy_pos, 'vel': noisy_vel}
        #Pursuers
        for i in range(self.num_pursuers):
            real_state['pursuers'].append(get_noisy_state(i))
        #Invaders
        for i in range(self.num_invaders):
            idx = self.num_pursuers + i
            real_state['invaders'].append(get_noisy_state(idx))
        #Prime
        prime_idx = self.total_drones - 1
        real_state['prime'] = get_noisy_state(prime_idx)
        return True, real_state

    def _set_color(self, idx, color):
        #only when the color is new
        if self.current_colors[idx] != color:
            p.changeVisualShape(self.env.DRONE_IDS[idx], -1, rgbaColor=color, physicsClientId=self.env.CLIENT)
            self.current_colors[idx] = color

    def close(self):
        if self.is_open:
            self.env.close()
            self.is_open = False