'''
Ego vehicle control
'''

import traci
import os
import random 
import csv

# Set the SUMO_HOME environment variable to your SUMO installation
os.environ["SUMO_HOME"] = "/opt/homebrew/opt/sumo/share/sumo"  # Adjust this path based on your system

# Start the SUMO simulation
sumo_binary = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo-gui")  # Or "sumo" for no-gui
sumo_config = "intersection.sumocfg"  # Your configuration file

# Start the simulation
# traci.start([sumo_binary, "-c", sumo_config]) 
# traci.start([sumo_binary, "-c", sumo_config, "--collision.check-junctions", "--collision.action", "warn","--random=true","--time-to-impatience=1"])
traci.start([sumo_binary, "-c", sumo_config, "--collision.check-junctions", "--collision.action", "warn","--random=true"])

# Run the simulation and control the ego vehicle's acceleration
step = 0

reward = 0
rewards = []
states = []
actions = []

while True:  # Run simulation until the ego vehicle passes the intersection or collides
    traci.simulationStep()  # Advance the simulation by one step
    step += 1

    # Define the vehicle ID for the ego vehicle
    ego_vehicle_id = "ego.0"

    # Check if the ego vehicle is in the simulation
    if ego_vehicle_id in traci.vehicle.getIDList():
        # Checks off all default speed mode settings
        traci.vehicle.setSpeedMode(ego_vehicle_id, 32)

        # Generate an arbitrary acceleration value `a` between -3 m/s² and 3 m/s²
        a = random.uniform(-3.0, 3.0) 

        # Finish the simulation if the ego vehicle has passed the intersection
        if traci.vehicle.getPosition(ego_vehicle_id)[0] > 60:
            print("Ego vehicle successfully passed the intersection!")
            break

        # Apply the acceleration to the ego vehicle
        traci.vehicle.setAcceleration(ego_vehicle_id, a, 1)

        speed = traci.vehicle.getSpeed(ego_vehicle_id)
        position = traci.vehicle.getPosition(ego_vehicle_id)

        collision = traci.simulation.getCollidingVehiclesIDList()
        if ego_vehicle_id in collision:
            print("Warning: Ego vehicle collided!")
            break

        # Optionally, print the applied acceleration value for debugging
        print(f"Step {step}: Applied acceleration {a:.2f} m/s² to {ego_vehicle_id}\nSpeed:{speed}\nPosition:{position}")

        state = [] 
        state.append((traci.vehicle.getPosition(ego_vehicle_id), traci.vehicle.getSpeed(ego_vehicle_id)))
        #state = (position, velocity, maximum accel deccel)
        for vehicle in traci.vehicle.getIDList():
            if vehicle != ego_vehicle_id:
                state.append((vehicle,
                              traci.vehicle.getPosition(vehicle), 
                              traci.vehicle.getSpeed(vehicle), 
                              traci.vehicle.getAcceleration(vehicle),
                              traci.vehicle.getAccel(vehicle)))

        states.append(state)
        actions.append(a)
        rewards.append(reward)

# Create a CSV file for states, action, and rewards
with open('rollout.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    for i in range(len(states)):
        # Write a single row to the CSV file
        writer.writerow([states[i], actions[i], rewards[i]])

traci.close()  # Close the simulation when done