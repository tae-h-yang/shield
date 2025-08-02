import traci
import os
import torch
import torch.nn as nn
import numpy as np
import traci.constants
import random
import math

# TODO is the agent not taking random action?

# Constants
MAX_STATE_FOE_NUMBER = 20
DISTANCE_THRESH = 58.0  # Distance threshold for ego vehicle to pass intersection

# Define the neural network for the DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64) 
        self.fc2 = nn.Linear(64, 128)          
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, actions, batch_size=128, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.actions = actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate

        # Create Q-network and target network
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)

        # Copy initial weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

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

def run_simulation(agent):
    step = 0
    state = None

    state_foe_ids = []

    speeding = 0
    stopping = 0
    emer_stop = 0

    while True:
        traci.simulationStep()
        step += 1
        ego_vehicle_id = "ego.0"

        if ego_vehicle_id in traci.vehicle.getIDList():
            # Collect ego vehicle state
            state = [step / 10,
                     round(traci.vehicle.getPosition(ego_vehicle_id)[0], 1),
                     round(traci.vehicle.getPosition(ego_vehicle_id)[1], 1),
                     round(traci.vehicle.getSpeed(ego_vehicle_id), 1),
                     round(traci.vehicle.getAcceleration(ego_vehicle_id), 1)]
            
            # Collect foe vehicle states
            foe_ids = list(traci.vehicle.getIDList())
            foe_ids.remove(ego_vehicle_id)
            # Store the foe ids in the order of spawn up to 10.
            if len(state_foe_ids) < MAX_STATE_FOE_NUMBER:
                for foe_id in foe_ids:
                    if foe_id not in state_foe_ids:
                        if len(state_foe_ids) < MAX_STATE_FOE_NUMBER:
                            state_foe_ids.append(foe_id)

            for foe_id in state_foe_ids:
                if foe_id not in foe_ids:
                    state.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    state.extend([round(traci.vehicle.getPosition(foe_id)[0], 1),
                                  round(traci.vehicle.getPosition(foe_id)[1], 1),
                                  round(traci.vehicle.getSpeed(foe_id), 1),
                                  round(traci.vehicle.getAcceleration(foe_id), 1),
                                  traci.vehicle.getImpatience(foe_id)])
            
            while len(state) < 5 + 5 * MAX_STATE_FOE_NUMBER:
                state.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            # print(state)

            traci.vehicle.setSpeedMode(ego_vehicle_id, 32)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_index = agent.q_network(state).argmax().item()
            action_value = agent.actions[action_index]
            traci.vehicle.setAcceleration(ego_vehicle_id, action_value, 100)

            max_speed = traci.lane.getMaxSpeed(traci.vehicle.getLaneID(ego_vehicle_id))
            ego_speed = traci.vehicle.getSpeed(ego_vehicle_id)
            if ego_speed > max_speed:
                speeding += 1

            ego_x = traci.vehicle.getPosition(ego_vehicle_id)[0]
            if ego_speed < 0.01 and (ego_x < 40 or ego_x > 45):
                stopping += 1

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

def load_trained_agent(agent, model_path="dqn_agent.pth"):
    # Load the saved model
    checkpoint = torch.load(model_path)
    agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
    agent.epsilon = checkpoint.get('epsilon', 0.01)
    print(f"Trained agent loaded from '{model_path}'.")

def evaluate_agent(agent, num_episodes=5, visualize=False):
    success, success_time, fail, speeding, stopping, emer_stop = 0, 0, 0, 0, 0, 0
    for episode in range(num_episodes):
        mode = "sumo-gui" if visualize else "sumo"
        start_simulation(mode)
        s, t, f, sp, p, em = run_simulation(agent)
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
    actions = [a for a in range(-3, 4)]
    # actions = [-4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 2.6]

    state_size = 5 + MAX_STATE_FOE_NUMBER * 5
    action_size = len(actions)
    agent = DQNAgent(state_size, action_size, actions)

    load_trained_agent(agent, model_path="dqn_agent.pth")

    # num_episodes = 500
    # success, average_success_time, fail, speeding, stopping, emer_stop = evaluate_agent(agent, num_episodes, visualize=False)
    # print("Number of successful passes: ", success)
    # print("Average time spent on successful passes: ", average_success_time)
    # print("Number of collisions: ", fail)
    # print("Average number of speedings per episode: ", speeding/num_episodes)
    # print("Average number of stoppings per episode: ", stopping/num_episodes)
    # print("Average number of emergency stoppings from road users: ", emer_stop/num_episodes)

    evaluate_agent(agent, num_episodes=5, visualize=True)

if __name__ == "__main__":
    main()
