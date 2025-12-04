from sim_config3D import Sim3DConfig
from simulator import DroneSimulation

def run():
    sc = Sim3DConfig()
    sim = DroneSimulation(sc, _3d=True)
    sim.run()
    
if __name__ == "__main__":
    run()
