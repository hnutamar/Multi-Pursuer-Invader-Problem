from sim_config3D import Sim3DConfig
from simulator import DroneSimulation

if __name__ == "__main__":
    sc = Sim3DConfig()
    sim = DroneSimulation(sc, _3d=True)
    sim.run()
