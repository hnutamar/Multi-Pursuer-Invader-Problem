from sim_config2D import Sim2DConfig
from simulator import DroneSimulation
from prime_mode import Modes

def run():
    sc = Sim2DConfig(purs_num=15, inv_num=0, obstacle=True, obstacle_rad=4.0, obstacle_pos=[15.0, 15.0])
    sim = DroneSimulation(sc, _3d=False, crash_enabled=1, purs_acc=3.0, inv_acc=4.5, prime_acc=1.2, 
                          prime_mode=Modes.LINE, inv_control=False, formation_delay=10)
    sim.run()

if __name__ == "__main__":
    run()