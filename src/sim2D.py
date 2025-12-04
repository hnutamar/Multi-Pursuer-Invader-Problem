from sim_config2D import Sim2DConfig
from simulator import DroneSimulation

def run():
    sc = Sim2DConfig()
    sim = DroneSimulation(sc, _3d=False)
    sim.run()

if __name__ == "__main__":
    run()