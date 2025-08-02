import traci  # Library to interact with SUMO
import os

# Set the SUMO_HOME environment variable to your SUMO installation
os.environ["SUMO_HOME"] = "/opt/homebrew/opt/sumo/share/sumo"  # or "/usr/local/sumo"

# Start the SUMO simulation
sumo_binary = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo-gui")  # or use "sumo" for no-gui
sumo_config = "intersection.sumocfg"  # Your configuration file

# Start the simulation
traci.start([sumo_binary, "-c", sumo_config])

step = 0
while step < 1000:  # Run simulation for 1000 steps
    traci.simulationStep()  # Move the simulation forward one step
    step += 1

    # Get the list of all vehicles in the simulation
    vehicles = traci.vehicle.getIDList()

    for vehicle in vehicles:
        # Get the position of the vehicle (x, y) in meters
        x, y = traci.vehicle.getPosition(vehicle)
        print(vehicle)

        if vehicle == "ego.0":
            print(x, y)
        
        # Define the intersection coordinates, for example:
        # Assuming intersection is at x=0, y=0 (adjust to actual coordinates)
        intersection_x = 100
        intersection_y = 100
        
        # Check if the vehicle is close enough to the intersection
        # Adjust the threshold (e.g., 10 meters) based on your setup
        if abs(x - intersection_x) < 10 and abs(y - intersection_y) < 10:
            traci.vehicle.setSpeed(vehicle, 0)  # Stop the vehicle when it reaches the intersection
            print(f"Vehicle {vehicle} stopped at the intersection.")
        else:
            traci.vehicle.setSpeed(vehicle, 15)  # Set the vehicle speed to 15 m/s when not at the intersection

traci.close()  # Close the simulation when done
