import traci
import os
import random
import numpy as np
import csv

# Constants
MAX_STATE_FOE_NUMBER = 10
DISTANCE_THRESH = 60  # Distance threshold for ego vehicle to pass intersection

def start_simulation():
    # Set the SUMO_HOME environment variable to your SUMO installation
    os.environ["SUMO_HOME"] = "/opt/homebrew/opt/sumo/share/sumo"  # Adjust this path based on your system

    # Start the SUMO simulation
    sumo_binary = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo-gui")
    sumo_config = "./sumo_env/intersection.sumocfg"  # Your configuration file

    # Start the simulation
    traci.start([sumo_binary, "-c", sumo_config, "--collision.check-junctions", "--collision.action", "warn", "--random=true"])

def close_simulation():
    traci.close()

def run_simulation():
    step = 0
    state_foe_ids = []
    # actions = [a for a in np.arange(-4.0, 3.5, 0.5)]  # Available actions for acceleration
    actions = [a for a in range(-3, 4)]

    state = None
    past_state = None
    action = None
    reward = None

    # Store state, action, reward, next_state pairs
    experience_list = []

    while True:  # Run simulation until the ego vehicle passes the intersection or collides
        traci.simulationStep()  # Advance the simulation by one step
        step += 1

        # Find the vehicle ID for the ego vehicle
        ego_vehicle_id = "ego.0"

        if ego_vehicle_id in traci.vehicle.getIDList():
            past_state = state  # Store the previous state

            # Collect state for the ego vehicle
            state = [step / 10,
                     round(traci.vehicle.getPosition(ego_vehicle_id)[0], 1),
                     round(traci.vehicle.getPosition(ego_vehicle_id)[1], 1),
                     round(traci.vehicle.getSpeed(ego_vehicle_id), 1),
                     round(traci.vehicle.getAcceleration(ego_vehicle_id), 1)]
            
            # Get list of foe vehicles (excluding the ego vehicle)
            foe_ids = list(traci.vehicle.getIDList())
            foe_ids.remove(ego_vehicle_id)

            # Store the foe ids in the order of spawn up to 10 foes
            if len(state_foe_ids) < MAX_STATE_FOE_NUMBER:
                for foe_id in foe_ids:
                    if foe_id not in state_foe_ids:
                        state_foe_ids.append(foe_id)

            # Add state information for each foe vehicle
            for foe_id in state_foe_ids:
                if foe_id not in foe_ids:
                    state.extend([0.0, 0.0, 0.0, 0.0])  # Add zeroes if foe is not in the simulation
                else:
                    state.extend([round(traci.vehicle.getPosition(foe_id)[0], 1),
                                  round(traci.vehicle.getPosition(foe_id)[1], 1),
                                  round(traci.vehicle.getSpeed(foe_id), 1),
                                  round(traci.vehicle.getAcceleration(foe_id), 1),
                                  traci.vehicle.getImpatience(foe_id)])

            # Ensure state has a fixed size (55) by padding with zeros if fewer than 10 foes
            while len(state) < 5 + 5 * MAX_STATE_FOE_NUMBER:  # 5 for ego vehicle + 5 * 10 for foes
                state.extend([0.0, 0.0, 0.0, 0.0, 0.0])  # Pad with zeros for missing foes

            # Checks off all default speed mode settings
            traci.vehicle.setSpeedMode(ego_vehicle_id, 32)

            # Choose an action from the list of available actions
            action = random.choice(actions)

            # Apply the acceleration to the ego vehicle
            traci.vehicle.setAcceleration(ego_vehicle_id, action, 100)

            # Reward calculation based on various conditions
            reward = 0
            reward += -0.1 * step / 10  # Penalize for each time step

            max_speed = traci.lane.getMaxSpeed(traci.vehicle.getLaneID(ego_vehicle_id))
            ego_speed = traci.vehicle.getSpeed(ego_vehicle_id)
            if ego_speed > max_speed:
                reward += 0.1 * (max_speed - ego_speed)  # Penalize for exceeding speed limit

            # If the ego vehicle passes the intersection, end the simulation with a positive reward
            if traci.vehicle.getPosition(ego_vehicle_id)[0] > DISTANCE_THRESH:
                reward += 10
                print("Ego vehicle successfully passed the intersection!")
                next_state = None  # No more states after the terminal state
                experience_list.append((past_state, action, reward, next_state))
                break

            # If a collision occurs, penalize heavily and end the simulation
            collision = traci.simulation.getCollidingVehiclesIDList()
            if ego_vehicle_id in collision:
                reward -= 100
                print("Warning: Ego vehicle collided!")
                next_state = None  # No more states after the terminal state
                experience_list.append((past_state, action, reward, next_state))
                break

            # Add (past_state, action, reward, next_state) to the experience list
            next_state = state  # Current state becomes the next state
            experience_list.append((past_state, action, reward, next_state))

        else:
            continue

    # # Optionally, save the experience list to a CSV for analysis
    # with open('experience_data.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     for experience in experience_list:
    #         writer.writerow(experience)

if __name__ == "__main__":
    iteration = 1
    for i in range(iteration):
        start_simulation()
        run_simulation()
        close_simulation()
