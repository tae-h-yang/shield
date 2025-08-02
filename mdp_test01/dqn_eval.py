import traci
import os
import torch
import torch.nn as nn
import numpy as np

# Constants
MAX_STATE_FOE_NUMBER = 10
DISTANCE_THRESH = 60  # Distance threshold for ego vehicle to pass intersection

# Define the neural network for the DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ReplayBuffer (dummy implementation for compatibility)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, actions, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001):
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

def run_simulation(agent):
    step = 0
    state = None

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

            for i in range(MAX_STATE_FOE_NUMBER):
                if i < len(foe_ids):
                    foe_id = foe_ids[i]
                    state.extend([round(traci.vehicle.getPosition(foe_id)[0], 1),
                                  round(traci.vehicle.getPosition(foe_id)[1], 1),
                                  round(traci.vehicle.getSpeed(foe_id), 1),
                                  round(traci.vehicle.getAcceleration(foe_id), 1),
                                  traci.vehicle.getImpatience(foe_id)])
                else:
                    state.extend([0, 0, 0, 0, 0])  # Padding

            print(state)

            traci.vehicle.setSpeedMode(ego_vehicle_id, 32)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_index = agent.q_network(state).argmax().item()
            action_value = agent.actions[action_index]
            traci.vehicle.setAcceleration(ego_vehicle_id, action_value, 100)

            if traci.vehicle.getPosition(ego_vehicle_id)[0] > DISTANCE_THRESH:
                print(f"Episode completed. Ego vehicle passed the intersection at step {step}.")
                break

            collision = traci.simulation.getCollidingVehiclesIDList()
            if ego_vehicle_id in collision:
                print(f"Collision occurred at step {step}.")
                break

def load_trained_agent(agent, model_path="dqn_agent.pth"):
    # Load the saved model
    checkpoint = torch.load(model_path)
    agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    agent.target_network.load_state_dict(checkpoint['target_network_state_dict'])
    agent.epsilon = checkpoint.get('epsilon', 0.01)
    print(f"Trained agent loaded from '{model_path}'.")

def evaluate_agent(agent, num_episodes=5, visualize=False):
    for episode in range(num_episodes):
        mode = "sumo-gui" if visualize else "sumo"
        start_simulation(mode)
        run_simulation(agent)
        close_simulation()
        print(f"Evaluation Episode {episode + 1}/{num_episodes} completed.")

def main():
    actions = [a for a in range(-4, 4)]
    # actions = [-4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 2.6]

    state_size = 5 + MAX_STATE_FOE_NUMBER * 5
    action_size = len(actions)
    agent = DQNAgent(state_size, action_size, actions)

    load_trained_agent(agent, model_path="dqn_agent.pth")
    evaluate_agent(agent, num_episodes=5, visualize=True)

if __name__ == "__main__":
    main()
