from sim_config3D import Sim3DConfig
from simulator import DroneSimulation

def run():
    sc = Sim3DConfig(purs_num=20, inv_num=2)
    sim = DroneSimulation(sc, _3d=True, purs_acc=3.8, inv_acc=0.5, formation_delay=10, crash_enabled=1)
    sim.run()
    
if __name__ == "__main__":
    run()
