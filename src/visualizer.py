import matplotlib.pyplot as plt
import numpy as np
from pursuer_states import States

class MatplotlibVisualizer:
    def __init__(self, sc_config, _3d=False):
        self.sc = sc_config
        self._3d = _3d
        self.fig = self.sc.fig
        self.ax = self.sc.ax
        
        self.is_open = True
        
        # Inicializace grafiky (převezmeme to, co už máš v sc_config, nebo nastavíme zde)
        # Předpokládám, že sc_config už má vytvořené objekty p_dots, i_dots atd.
        
        # Manuální ovládání - stav kláves
        self.manual_vel = np.zeros(3) if self._3d else np.zeros(2)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self._on_key_release)
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        
        plt.ion()
        plt.show(block=False)

    def render(self, state, world_instance=None, quiver=False):
        """Aktualizuje grafiku na základě stavu."""
        
        # 1. Pursuers
        for i, (p_pos, p_state) in enumerate(zip(state['pursuers'], state['pursuers_status'])):
            p_dot = self.sc.p_dots[i]
            p_dot.set_data([p_pos[0]], [p_pos[1]])
            if self._3d:
                p_dot.set_3d_properties([p_pos[2]])
            
            # Barva podle stavu
            p_dot.set_color("#631616" if p_state == States.PURSUE else '#d62728')

        # 2. Invaders
        for i, i_pos in enumerate(state['invaders']):
            i_dot = self.sc.i_dots[i]
            i_dot.set_data([i_pos[0]], [i_pos[1]])
            if self._3d:
                i_dot.set_3d_properties([i_pos[2]])

        # 3. Prime
        u_pos = state['prime']
        self.sc.u_dot.set_data([u_pos[0]], [u_pos[1]])
        if self._3d:
            self.sc.u_dot.set_3d_properties([u_pos[2]])
            
        # 4. Vortex Field (Quiver) - pokud chceme
        if not self._3d and world_instance is not None and quiver:
             self._update_vector_field(world_instance)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _update_vector_field(self, world):
        # Tady zkopíruj svou logiku _update_vector_field
        # Ale pozor: místo self.pursuers[0] použij world.pursuers[0]
        #making quiver graph, visualizing vortex field
        unit = world.prime
        center = unit.position
        grid_x = self.sc.grid_x_base + center[0]
        grid_y = self.sc.grid_y_base + center[1]
        u_arrows = []
        v_arrows = []
        if not world.pursuers:
            return
        ref_pursuer = world.pursuers[0]
        flat_x = grid_x.flatten()
        flat_y = grid_y.flatten()
        for x, y in zip(flat_x, flat_y):
            test_pos = np.array([x, y])
            force_vec = ref_pursuer.form_vortex_field_circle(unit, mock_position=test_pos, obstacle=world.obstacle)
            norm = np.linalg.norm(force_vec)
            if norm > 0:
                force_vec = force_vec / norm
            u_arrows.append(force_vec[0])
            v_arrows.append(force_vec[1])
        new_offsets = np.column_stack((flat_x, flat_y))
        self.sc.quiver.set_offsets(new_offsets)
        self.sc.quiver.set_UVC(u_arrows, v_arrows)
        #quiver graph for invaders
        for idx, unit in enumerate(world.invaders):
            center = unit.position
            grid_x = self.sc.inv_grid_x_base[idx] + center[0]
            grid_y = self.sc.inv_grid_y_base[idx] + center[1]
            u_arrows = []
            v_arrows = []
            ref_pursuer = world.pursuers[0]
            flat_x = grid_x.flatten()
            flat_y = grid_y.flatten()
            for x, y in zip(flat_x, flat_y):
                test_pos = np.array([x, y])
                force_vec = ref_pursuer.pursuit_circling([unit, 0], mock_position=test_pos)
                norm = np.linalg.norm(force_vec)
                if norm > 0:
                    force_vec = force_vec / norm
                u_arrows.append(force_vec[0])
                v_arrows.append(force_vec[1])
            new_offsets = np.column_stack((flat_x, flat_y))
            self.sc.inv_quiver[idx].set_offsets(new_offsets)
            self.sc.inv_quiver[idx].set_UVC(u_arrows, v_arrows)

    def _on_close(self, event):
        self.is_open = False

    def _on_key_press(self, event):
        # Stejná logika jako dřív
        if event.key == 'up': self.manual_vel[1] = 1.0
        elif event.key == 'down': self.manual_vel[1] = -1.0
        elif event.key == 'left': self.manual_vel[0] = -1.0
        elif event.key == 'right': self.manual_vel[0] = 1.0
        if self._3d:
            if event.key == 'w': self.manual_vel[2] = 1.0
            elif event.key == 's': self.manual_vel[2] = -1.0

    def _on_key_release(self, event):
        # Stejná logika pro nulování rychlosti
        if event.key in ['up', 'down']: self.manual_vel[1] = 0.0
        elif event.key in ['left', 'right']: self.manual_vel[0] = 0.0
        if self._3d and event.key in ['w', 's']: self.manual_vel[2] = 0.0