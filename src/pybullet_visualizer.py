import pybullet as p
import pybullet_data
import numpy as np
from pursuer_states import States

class PyBulletVisualizer:
    def __init__(self, sc_config, _3d=False, quiver=False):
        self.sc = sc_config
        self._3d = _3d
        self.is_open = True
        #invader manual control
        self.manual_vel = np.zeros(3) if self._3d else np.zeros(2)
        #staring pybullet
        self.client = p.connect(p.GUI)
        #side pannels closed
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        #no shadows
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        #no gravity
        p.setGravity(0, 0, 0)
        #loading plane
        self.plane_id = p.loadURDF("plane.urdf")
        #ids
        self.p_ids = []
        self.i_ids = []
        self.obs_ids = []
        #cached color
        self.current_p_colors = [None] * self.sc.PURSUER_NUM
        #colors
        self.C_RED = [0.84, 0.15, 0.15, 1.0]
        self.C_BLUE = [0.12, 0.46, 0.7, 1.0]
        self.C_GREEN = [0.06, 0.92, 0.13, 1.0]
        self.C_BLACK = [0.0, 0.0, 0.0, 1.0]
        self.C_YELLOW = [1.0, 0.93, 0.0, 1.0]
        self.C_GREY = [0.3, 0.3, 0.3, 1.0]
        #drone models path
        urdf_path = "assets/cf2x.urdf" 
        #bigger scale
        drone_scale = 10.0 
        #flag for loading
        fast_load_flag = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        #pursuers
        for i in range(self.sc.PURSUER_NUM):
            start_pos = [i * 0.2, 0, 2] 
            drone_id = p.loadURDF(urdf_path, basePosition=start_pos, 
                                  globalScaling=drone_scale, flags=fast_load_flag)
            self.p_ids.append(drone_id)
            p.changeVisualShape(drone_id, -1, rgbaColor=self.C_RED)
            self.current_p_colors[i] = self.C_RED    
        #invaders
        for i in range(self.sc.INVADER_NUM):
            start_pos = [i * 0.2, 5, 2]
            drone_id = p.loadURDF(urdf_path, basePosition=start_pos, 
                                  globalScaling=drone_scale, flags=fast_load_flag)
            self.i_ids.append(drone_id)
            p.changeVisualShape(drone_id, -1, rgbaColor=self.C_BLUE)
        #prime
        self.prime_id = p.loadURDF(urdf_path, basePosition=[0,0,5], 
                                   globalScaling=drone_scale * 1.5, flags=fast_load_flag)
        p.changeVisualShape(self.prime_id, -1, rgbaColor=self.C_GREEN)
        #obstacles
        if self.sc.obstacle:
            for i in range(len(self.sc.obs_rads)):
                pos = self.sc.obs_pos[i]
                rad = self.sc.obs_rads[i]
                pos_3d = list(pos) if self._3d else [pos[0], pos[1], 1.0]
                
                visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=rad, rgbaColor=self.C_GREY)
                obs_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_id, basePosition=pos_3d)
                self.obs_ids.append(obs_id)
        #camera pos
        p.resetDebugVisualizerCamera(cameraDistance=30, cameraYaw=45, cameraPitch=-35, 
                                     cameraTargetPosition=[self.sc.WORLD_WIDTH/2, self.sc.WORLD_HEIGHT/2, 0])

    def _create_sphere(self, radius, color, pos=[0, 0, 0]):
        visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        #mass 0, kinematic body
        body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_id, basePosition=pos)
        return body_id

    def render(self, state, world_instance=None):
        if not self.is_open or not p.isConnected():
            self.is_open = False
            return False
        #if ESC was pressed, close animation
        quit_requested = self._handle_keyboard()
        if quit_requested:
            self.is_open = False
            return False
        #Prime
        pos = state['prime']
        pos_3d = list(pos) if self._3d else [pos[0], pos[1], 1.0]
        p.resetBasePositionAndOrientation(self.prime_id, pos_3d, [0, 0, 0, 1])
        #Invaders
        for i, pos in enumerate(state['invaders']):
            pos_3d = list(pos) if self._3d else [pos[0], pos[1], 1.0]
            p.resetBasePositionAndOrientation(self.i_ids[i], pos_3d, [0, 0, 0, 1])
        #Pursuers
        for i, (pos, p_state) in enumerate(zip(state['pursuers'], state['pursuers_status'])):
            pos_3d = list(pos) if self._3d else [pos[0], pos[1], 1.0]
            p.resetBasePositionAndOrientation(self.p_ids[i], pos_3d, [0, 0, 0, 1])
            #color change
            target_color = self.C_RED
            if p_state[0] == States.PURSUE:
                if p_state[1][1] == 1:
                    target_color = self.C_BLACK
                else:
                    target_color = self.C_YELLOW
            #change it only if it is new, saves a lot of compute power
            if self.current_p_colors[i] != target_color:
                p.changeVisualShape(self.p_ids[i], -1, rgbaColor=target_color)
                self.current_p_colors[i] = target_color
        #pybullet step
        p.stepSimulation()
        return True

    def _handle_keyboard(self):
        #keyboard events
        keys = p.getKeyboardEvents()
        #ESC
        if 27 in keys and (keys[27] & p.KEY_IS_DOWN):
            return True
        #manual control
        self.manual_vel[:] = 0.0
        if ord('w') in keys and (keys[ord('w')] & p.KEY_IS_DOWN): self.manual_vel[1] = 1.0
        if ord('s') in keys and (keys[ord('s')] & p.KEY_IS_DOWN): self.manual_vel[1] = -1.0
        if ord('a') in keys and (keys[ord('a')] & p.KEY_IS_DOWN): self.manual_vel[0] = -1.0
        if ord('d') in keys and (keys[ord('d')] & p.KEY_IS_DOWN): self.manual_vel[0] = 1.0
        
        if self._3d:
            if ord('q') in keys and (keys[ord('q')] & p.KEY_IS_DOWN): self.manual_vel[2] = 1.0
            if ord('e') in keys and (keys[ord('e')] & p.KEY_IS_DOWN): self.manual_vel[2] = -1.0
            
        return False
    
    def close(self):
        #disconnetct from the pybullet client
        if hasattr(self, 'client'):
            try:
                p.disconnect(self.client)
            except:
                pass
        self.is_open = False