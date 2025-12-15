from sim_config2D import Sim2DConfig
from simulator import DroneSimulation
from prime_mode import Modes

def run():
    sc = Sim2DConfig(purs_num=20, inv_num=1)
    sim = DroneSimulation(sc, _3d=False, crash_enabled=1, purs_acc=1.7, inv_acc=3.0, prime_acc=0.2, 
                          prime_mode=Modes.CIRCLE, inv_control=True)
    sim.run()

if __name__ == "__main__":
    run()