import traci
import os
import torch
import torch.nn as nn
import numpy as np
import traci.constants
import random
import math

DISTANCE_THRESH = 58.0

# Start SUMO simulation
def start_simulation(mode="sumo"):
    os.environ["SUMO_HOME"] = "/opt/homebrew/opt/sumo/share/sumo"  # Adjust path to your SUMO installation
    sumo_binary = os.path.join(os.environ["SUMO_HOME"], "bin", mode)
    sumo_config = "./sumo_env/intersection.sumocfg"  # Update with your config file path
    traci.start([sumo_binary, "-c", sumo_config, "--collision.check-junctions", "--collision.action", "warn", "--random=true"])

def close_simulation():
    traci.close()


def get_distance(ego_id, foe_id):
    ego_pos = traci.vehicle.getPosition(ego_id)
    foe_pos = traci.vehicle.getPosition(foe_id)
    return math.sqrt((ego_pos[0]-foe_pos[0])**2 + (ego_pos[1]-foe_pos[1])**2)

def run_simulation():
    step = 0
    
    actions = [a for a in range(-3, 4)]

    speeding = 0
    stopping = 0
    emer_stop = 0

    while True:
        traci.simulationStep()
        step += 1
        ego_vehicle_id = "ego.0"

        if ego_vehicle_id in traci.vehicle.getIDList():
            # print(state)

            traci.vehicle.setSpeedMode(ego_vehicle_id, 32)
            a = random.choice(actions)
            traci.vehicle.setAcceleration(ego_vehicle_id, a, 100)

            max_speed = traci.lane.getMaxSpeed(traci.vehicle.getLaneID(ego_vehicle_id))
            ego_speed = traci.vehicle.getSpeed(ego_vehicle_id)
            if ego_speed > max_speed:
                speeding += 1

            ego_x = traci.vehicle.getPosition(ego_vehicle_id)[0]
            if ego_speed < 0.01 and (ego_x < 40 or ego_x > 45):
                stopping += 1

            foe_ids = list(traci.vehicle.getIDList())
            foe_ids.remove(ego_vehicle_id)

            for foe_id in foe_ids:
                if get_distance(ego_vehicle_id, foe_id) < 16:
                    if traci.vehicle.getAcceleration(foe_id) < -8:
                        emer_stop += 1

            if traci.vehicle.getPosition(ego_vehicle_id)[0] > DISTANCE_THRESH:
                print(f"Episode completed. Ego vehicle passed the intersection at step {step}.")
                return 1, step, 0, speeding, stopping, emer_stop

            collision = traci.simulation.getCollidingVehiclesIDList()
            if ego_vehicle_id in collision:
                print(f"Collision occurred at step {step}.")
                return 0, 0, 1, speeding, stopping, emer_stop

def evaluate_random_policy(num_episodes=5, visualize=False):
    success, success_time, fail, speeding, stopping, emer_stop = 0, 0, 0, 0, 0, 0
    for episode in range(num_episodes):
        mode = "sumo-gui" if visualize else "sumo"
        start_simulation(mode)
        s, t, f, sp, p, em = run_simulation()
        close_simulation()
        print(f"Evaluation Episode {episode + 1}/{num_episodes} completed.")
        success += s
        success_time += t
        fail += f
        speeding += sp
        stopping += p
        emer_stop += em
    return success, success_time/success, fail, speeding, stopping, emer_stop

def main():
    num_episodes = 500
    success, average_success_time, fail, speeding, stopping, emer_stop = evaluate_random_policy(num_episodes, visualize=False)
    print("Number of successful passes: ", success)
    print("Average time spent on successful passes: ", average_success_time)
    print("Number of collisions: ", fail)
    print("Average number of speedings per episode: ", speeding/num_episodes)
    print("Average number of stoppings per episode: ", stopping/num_episodes)
    print("Average number of emergency stoppings from road users: ", emer_stop/num_episodes)

    # evaluate_random_policy(num_episodes=5, visualize=True)

if __name__ == "__main__":
    main()
