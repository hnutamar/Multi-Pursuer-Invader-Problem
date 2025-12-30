from sim_config2D import Sim2DConfig
from simulator import DroneSimulation
from prime_mode import Modes

def run():
    sc = Sim2DConfig(purs_num=10, inv_num=0, obstacle=True)
    sim = DroneSimulation(sc, _3d=False, crash_enabled=1, purs_acc=3.0, inv_acc=4.5, prime_acc=0.3, 
                          prime_mode=Modes.LINE, inv_control=True, formation_delay=10)
    sim.run()

if __name__ == "__main__":
    run()