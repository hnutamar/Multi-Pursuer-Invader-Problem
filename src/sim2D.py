from sim_config2D import Sim2DConfig
from simulator import DroneSimulation
from prime_mode import Modes

def run():
    sc = Sim2DConfig(purs_num=30, inv_num=0)
    sim = DroneSimulation(sc, _3d=False, crash_enabled=1, purs_acc=3.0, inv_acc=1.5, prime_acc=0.1, 
                          prime_mode=Modes.CIRCLE, inv_control=False)
    sim.run()

if __name__ == "__main__":
    run()