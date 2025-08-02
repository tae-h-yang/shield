import traci  # Library to interact with SUMO
import os

# Set the SUMO_HOME environment variable to your SUMO installation
os.environ["SUMO_HOME"] = "/opt/homebrew/opt/sumo/share/sumo"  # or "/usr/local/sumo"

# Start the SUMO simulation
sumo_binary = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo-gui")  # or use "sumo" for no-gui
sumo_config = "intersection.sumocfg"

# Start the simulation
traci.start([sumo_binary, "-c", sumo_config])

step = 0
while step < 1000:  # Run simulation for 1000 steps
    traci.simulationStep()  # Move the simulation forward one step
    step += 1

    # You can add real-time control here (e.g., change vehicle speed, flow, or routes)
    vehicles = traci.vehicle.getIDList()
    for vehicle in vehicles:
        if step % 10 == 0:
            traci.vehicle.setSpeed(vehicle, 15)  # Change speed of vehicles every 10 steps

traci.close()  # Close the simulation when done
