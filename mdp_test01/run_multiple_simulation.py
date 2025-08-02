'''
MDP Solver
The state space considers up to 10 vehicles from the beginning assumming 
that the 10 vehicles are sufficient for the ego vehicle to consider until passing the intersection safely.
'''
import traci
import os
import random 
import numpy as np
import csv

def start_simulation():
    # Set the SUMO_HOME environment variable to your SUMO installation
    os.environ["SUMO_HOME"] = "/opt/homebrew/opt/sumo/share/sumo"  # Adjust this path based on your system

    # Start the SUMO simulation
    sumo_binary = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo-gui")
    # sumo_binary = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo")  # Or "sumo" for no-gui
    # Make sure to run "netconvert -c intersection.netccfg" to generate the latest simulation setting.
    sumo_config = "./sumo_env/intersection.sumocfg"  # Your configuration file

    # Start the simulation
    # traci.start([sumo_binary, "-c", sumo_config]) 
    # traci.start([sumo_binary, "-c", sumo_config, "--collision.check-junctions", "--collision.action", "warn","--random=true","--time-to-impatience=1"])
    traci.start([sumo_binary, "-c", sumo_config, "--collision.check-junctions", "--collision.action", "warn","--random=true"])

def close_simulation():
    traci.close() 

def run_simulation():
    # Run the simulation and control the ego vehicle's acceleration
    step = 0

    # reward = 0
    # rewards = []
    # states = []
    # actions = []
    MAX_STATE_FOE_NUMBER = 10
    state_foe_ids = []

    actions = [a for a in np.arange(-4.0, 3.5, 0.5)]
    # print(actions)
    DISTANCE_THRESH = 60

    while True:  # Run simulation until the ego vehicle passes the intersection or collides
        traci.simulationStep()  # Advance the simulation by one step
        step += 1

        # Find the vehicle ID for the ego vehicle according to the "flow id" in "intersection.rou.xml"
        ego_vehicle_id = "ego.0"

        # print("Traci time: ", traci.simulation.getTime(), " ", step/10) 

        # Check if the ego vehicle is in the simulation
        if ego_vehicle_id in traci.vehicle.getIDList():
            # TODO 
            # add max accel of foe vehicles using getAccel()?
            # add ego_accel as it's not always the same as the accel commanded to ego.
            # State = [time, ego_pos_x, ego_pos_y, ego_speed, ego_accel, foe_pos_x, foe_pos_y, foe_speed, foe_accel, foe_patience, ...]
            state = [step/10,
                     round(traci.vehicle.getPosition(ego_vehicle_id)[0], 1),
                     round(traci.vehicle.getPosition(ego_vehicle_id)[1], 1),
                     round(traci.vehicle.getSpeed(ego_vehicle_id), 1),
                     round(traci.vehicle.getAcceleration(ego_vehicle_id), 1)]
            
            # foe_ids are stored in alphabetical order
            foe_ids = list(traci.vehicle.getIDList())
            foe_ids.remove(ego_vehicle_id)
            # Store the foe ids in the order of spawn up to 10.
            if len(state_foe_ids) < MAX_STATE_FOE_NUMBER:
                for foe_id in foe_ids:
                    if foe_id not in state_foe_ids:
                        state_foe_ids.append(foe_id)
            # print(state_foe_ids)   

            for foe_id in state_foe_ids:
                if foe_id not in foe_ids:
                    state.extend([0.0, 0.0, 0.0, 0.0])
                else:
                    state.extend([round(traci.vehicle.getPosition(foe_id)[0], 1),
                                  round(traci.vehicle.getPosition(foe_id)[1], 1),
                                  round(traci.vehicle.getSpeed(foe_id), 1),
                                  round(traci.vehicle.getAcceleration(foe_id), 1),
                                  traci.vehicle.getImpatience(foe_id)])
            print(state)
                
            # Checks off all default speed mode settings
            traci.vehicle.setSpeedMode(ego_vehicle_id, 32)
            # Generate an arbitrary acceleration value `a` between -3 m/s² and 3 m/s²
            # a = random.uniform(-3.0, 3.0) 
            a = random.choice(actions)


            # Apply the acceleration to the ego vehicle
            # When the duration is 0.1, the actual acceleration value is a/2
            # traci.vehicle.setAcceleration(ego_vehicle_id, a, 0.1)
            # traci.vehicle.setAccel(ego_vehicle_id, 10)
            # When the duration is 100, the actual acceleartion is similar to the acceleration value commanded
            traci.vehicle.setAcceleration(ego_vehicle_id, a, 100)

            # a != traci.vehicle.getAcceleration(), ego's default accel limit is 2.6 and decel limit is 4.5
            # print("accel: ", a, traci.vehicle.getSpeed(ego_vehicle_id), traci.vehicle.getAcceleration(ego_vehicle_id))
            
            # Maximum speed limit is overriden by setAcceleration.
            # traci.vehicle.setMaxSpeed(ego_vehicle_id, 1)

            reward = 0

            reward += -0.1*step/10

            max_speed = traci.lane.getMaxSpeed(traci.vehicle.getLaneID(ego_vehicle_id))
            ego_speed = traci.vehicle.getSpeed(ego_vehicle_id)
            # print("max speed: ", max_speed)
            if ego_speed > max_speed:
                reward += 0.1*(max_speed - ego_speed)

            # Finish the simulation if the ego vehicle has passed the intersection
            if traci.vehicle.getPosition(ego_vehicle_id)[0] > DISTANCE_THRESH:
                reward += 10
                print("Ego vehicle successfully passed the intersection!")
                break

            collision = traci.simulation.getCollidingVehiclesIDList()
            if ego_vehicle_id in collision:
                reward -= 100
                print("Warning: Ego vehicle collided!")
                break

            # Optionally, print the applied acceleration value for debugging
            # print(f"Step {step}: Applied acceleration {a:.2f} m/s² to {ego_vehicle_id}\nSpeed:{speed}\nPosition:{position}")

            # state = [] 
            # state.append((traci.vehicle.getPosition(ego_vehicle_id), traci.vehicle.getSpeed(ego_vehicle_id)))
            #state = (position, velocity, maximum accel deccel)
            # for vehicle in traci.vehicle.getIDList():
            #     if vehicle != ego_vehicle_id:
            #         state.append((vehicle,
            #                     traci.vehicle.getPosition(vehicle), 
            #                     traci.vehicle.getSpeed(vehicle), 
            #                     traci.vehicle.getAcceleration(vehicle),
            #                     traci.vehicle.getAccel(vehicle)))

            # states.append(state)
            # actions.append(a)
            # rewards.append(reward)
        else:
            continue

    # # Create a CSV file for states, action, and rewards
    # with open('rollout.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file)
        
    #     for i in range(len(states)):
    #         # Write a single row to the CSV file
    #         writer.writerow([states[i], actions[i], rewards[i]])

if __name__ == "__main__":
    iteration = 1
    for i in range(iteration):
        start_simulation()
        run_simulation()
        close_simulation()