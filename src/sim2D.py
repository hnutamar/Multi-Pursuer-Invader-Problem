from sim_config2D import Sim2DConfig
from simulator import DroneSimulation

if __name__ == "__main__":
    sc = Sim2DConfig()
    sim = DroneSimulation(sc, _3d=False)
    sim.run()