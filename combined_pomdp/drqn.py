import traci
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import math

# import traci.check_constants
import traci.constants

# Constants
MAX_STATE_FOE_NUMBER = 20
DISTANCE_THRESH = 58.0  # Distance threshold for ego vehicle to pass intersection

class DRQN(nn.Module):
    def __init__(self, observation_size, action_size, hidden_size=64):
        super(DRQN, self).__init__()
        self.fc1 = nn.Linear(observation_size, hidden_size)  # Feature extraction from observations
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)  # Temporal modeling
        self.fc2 = nn.Linear(hidden_size, action_size)  # Q-values for actions

    def forward(self, x, hidden=None):
        x = torch.relu(self.fc1(x))
        x, hidden = self.lstm(x, hidden)  # LSTM expects 3D input (batch, seq_len, feature_size)
        x = self.fc2(x)
        return x, hidden
    
# # Define the neural network for the DRQN
# class DRQN(nn.Module):
#     def __init__(self, state_size, action_size, hidden_size=128, lstm_layers=1):
#         super(DRQN, self).__init__()
#         self.fc1 = nn.Linear(state_size, 64)  # Initial fully connected layer
#         self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True)
#         self.fc2 = nn.Linear(hidden_size, 64)  # Fully connected layer after LSTM
#         self.fc3 = nn.Linear(64, action_size)  # Output layer

#     def forward(self, x, hidden_state=None):
#         # Process the input through the initial fully connected layer
#         x = torch.relu(self.fc1(x))

#         # Add a batch dimension if needed for LSTM processing
#         x = x.unsqueeze(1) if x.ndim == 2 else x  # Ensure input has shape (batch, seq_len, features)

#         # Process through LSTM
#         if hidden_state is None:
#             x, hidden_state = self.lstm(x)  # Initialize hidden state internally
#         else:
#             x, hidden_state = self.lstm(x, hidden_state)  # Use provided hidden state

#         # Take the output of the LSTM's final time step
#         x = x[:, -1, :]  # Shape: (batch, hidden_size)

#         # Process through the fully connected layers
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x), hidden_state


class ReplayBuffer:
    def __init__(self, capacity, seq_length):
        self.capacity = capacity
        self.seq_length = seq_length
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        # Append a single experience tuple (obs, action, reward, next_obs, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        # Randomly sample batches of sequences
        indices = np.random.choice(len(self.buffer) - self.seq_length, batch_size)
        sequences = []
        for idx in indices:
            seq = list(self.buffer)[idx:idx + self.seq_length]
            sequences.append(seq)
        return sequences

    def size(self):
        return len(self.buffer)

class DRQNAgent:
    def __init__(self, observation_size, action_size, actions, seq_length=10, batch_size=128, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001):
        self.observation_size = observation_size
        self.action_size = action_size
        self.actions = actions
        self.seq_length = seq_length  # Sequence length for RNN training
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate

        # DRQN model
        self.q_network = DRQN(observation_size, action_size)
        self.target_network = DRQN(observation_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.memory = ReplayBuffer(10000, seq_length)

    def act(self, observation, hidden_state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, len(self.actions) - 1), hidden_state
        else:
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch and seq dim
            q_values, hidden_state = self.q_network(observation, hidden_state)
            return torch.argmax(q_values.squeeze(0)).item(), hidden_state

    def train(self):
        if self.memory.size() < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for seq in batch:
            s, a, r, ns, d = zip(*seq)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        # print(next_states)
        # print(len(next_states))
        # print(len(states))
        # print(len(states[0][0]))
        # print(len(next_states))
        # print(len(next_states[0]))
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute Q-values and targets
        q_values, _ = self.q_network(states)
        q_values = q_values.gather(2, actions.unsqueeze(2)).squeeze(2)

        with torch.no_grad():
            next_q_values, _ = self.target_network(next_states)
            max_next_q_values = next_q_values.max(2)[0]
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # Compute loss and update
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

def get_distance(ego_id, foe_id):
    ego_pos = traci.vehicle.getPosition(ego_id)
    foe_pos = traci.vehicle.getPosition(foe_id)
    return math.sqrt((ego_pos[0]-foe_pos[0])**2 + (ego_pos[1]-foe_pos[1])**2)

def start_simulation(mode="sumo"):
    os.environ["SUMO_HOME"] = "/opt/homebrew/opt/sumo/share/sumo"  # Adjust path based on your system
    # mode is either "sumo" or "sumo-gui"
    sumo_binary = os.path.join(os.environ["SUMO_HOME"], "bin", mode)
    sumo_config = "./sumo_env/intersection.sumocfg"  # Your SUMO configuration file
    traci.start([sumo_binary, "-c", sumo_config, "--collision.check-junctions", "--collision.action", "warn", "--random=true"])

def close_simulation():
    traci.close()

def run_simulation(agent):
    step = 0
    # Actions = [a for a in np.arange(-4.0, 3.5, 0.5)]  # Available actions for acceleration

    state = None
    past_state = None
    action = None
    reward = None

    state_foe_ids = []
    hidden_state = None  # Initialize the hidden state for LSTM

    while True:
        traci.simulationStep()
        step += 1
        ego_vehicle_id = "ego.0"

        if ego_vehicle_id in traci.vehicle.getIDList():
            # Store the previous state for transition
            past_state = state
            state = [step / 10,
                     round(traci.vehicle.getPosition(ego_vehicle_id)[0], 1),
                     round(traci.vehicle.getPosition(ego_vehicle_id)[1], 1),
                     round(traci.vehicle.getSpeed(ego_vehicle_id), 1),
                     round(traci.vehicle.getAcceleration(ego_vehicle_id), 1)]
            
            # Collect foe vehicle states
            foe_ids = list(traci.vehicle.getIDList())
            foe_ids.remove(ego_vehicle_id)

            # Add vehicles to the list of foe IDs, up to MAX_STATE_FOE_NUMBER
            if len(state_foe_ids) < MAX_STATE_FOE_NUMBER:
                for foe_id in foe_ids:
                    if foe_id not in state_foe_ids:
                        if len(state_foe_ids) < MAX_STATE_FOE_NUMBER:
                            state_foe_ids.append(foe_id)

            # Add foe vehicle information to the state
            for foe_id in state_foe_ids:
                if foe_id not in foe_ids:
                    state.extend([0.0, 0.0, 0.0, 0.0])  # Padding for missing foe vehicle
                else:
                    state.extend([round(traci.vehicle.getPosition(foe_id)[0], 1),
                                  round(traci.vehicle.getPosition(foe_id)[1], 1),
                                  round(traci.vehicle.getSpeed(foe_id), 1),
                                  round(traci.vehicle.getAcceleration(foe_id), 1)])
            
            while len(state) < 5 + 4 * MAX_STATE_FOE_NUMBER:
                state.extend([0.0, 0.0, 0.0, 0.0])  # Padding to match max foe count

            # Set the vehicle's speed mode (mode 32 for autonomous)
            traci.vehicle.setSpeedMode(ego_vehicle_id, 32)

            # Act using the DRQN agent, passing the hidden state for temporal consistency
            action_index, hidden_state = agent.act(state, hidden_state)
            action_value = agent.actions[action_index]  # Get the corresponding action value
            traci.vehicle.setAcceleration(ego_vehicle_id, action_value, 100)

            # Reward calculation logic
            reward = -0.5 * step / 10
            max_speed = traci.lane.getMaxSpeed(traci.vehicle.getLaneID(ego_vehicle_id))
            ego_speed = traci.vehicle.getSpeed(ego_vehicle_id)
            if ego_speed > max_speed:
                reward += 0.1 * (max_speed - ego_speed)

            # Penalize stopping far from the stop line
            ego_x = traci.vehicle.getPosition(ego_vehicle_id)[0]
            if ego_speed < 0.01 and (ego_x < 40 or ego_x > 45):
                print("Stopping at the place far from the stop line")
                reward -= 1

            # Add penalty for emergency braking (if applicable)
            # TODO: Add emergency braking logic (uncomment and implement when necessary)
            # emergency_stopping_vehicles = traci.simulation.getUniversal(traci.constants.VAR_EMERGENCYSTOPPING_VEHICLES_IDS)
            # if traci.simulation._getUniversal(traci.constants.VAR_EMERGENCYSTOPPING_VEHICLES_NUMBER) > 0:
            #     print("Emergency braking occurred!!")
            #     reward += -5

            # Penalty for too close to other vehicles
            for foe_id in foe_ids:
                if get_distance(ego_vehicle_id, foe_id) < 16:
                    if traci.vehicle.getAcceleration(foe_id) < -8:
                        reward -= 10

            # Add reward for distance traveled (optional)
            # TODO: Consider adding positive reward based on distance traveled.
            # print(len(state))
            # if past_state is not None:
            #     print(len(past_state))

            # Check if the vehicle has passed the threshold distance
            if traci.vehicle.getPosition(ego_vehicle_id)[0] > DISTANCE_THRESH:
                reward += 10
                agent.memory.push((state, action_index, reward, [0.0]*(5+4*MAX_STATE_FOE_NUMBER), 1))  # Done flag is 1 (episode ends)
                break

            # Collision check
            collision = traci.simulation.getCollidingVehiclesIDList()
            if ego_vehicle_id in collision:
                reward -= 100
                agent.memory.push((state, action_index, reward, [0.0]*(5+4*MAX_STATE_FOE_NUMBER), 1))  # End the episode due to collision
                break

            # Store the transition for training
            if past_state is not None:
                agent.memory.push((past_state, action_index, reward, state, 0))  # Done flag is 0 (not finished)

            # print(state)
            # print(len(state))
            # Train the agent
            agent.train()

def main():
    # Define the action space
    actions = [a for a in range(-3, 4)]  # Action values for acceleration
    state_size = 5 + MAX_STATE_FOE_NUMBER * 4  # Updated state size (without impatience)
    action_size = len(actions)

    # Initialize the DRQN agent
    agent = DRQNAgent(
        observation_size=state_size,
        action_size=action_size,
        actions=actions,
        seq_length=10,  # Define sequence length for training
        batch_size=64,  # Batch size for sequence-based training
    )

    iterations = 5000  # Number of episodes to train
    for episode in range(iterations):
        print(f"Starting Episode {episode + 1}/{iterations}")

        # Start the SUMO simulation
        start_simulation()

        # Run the simulation with the DRQN agent
        run_simulation(agent)

        # Close the SUMO simulation
        close_simulation()

        # Update the target network periodically
        if episode % 10 == 0:
            agent.update_target_network()
            print(f"Target network updated at episode {episode + 1}")

        # Save the agent periodically
        if (episode + 1) % 1000 == 0:
            torch.save({
                'q_network_state_dict': agent.q_network.state_dict(),
                'target_network_state_dict': agent.target_network.state_dict(),
                'epsilon': agent.epsilon
            }, f"drqn_agent_episode_{episode + 1}.pth")
            print(f"Model saved at episode {episode + 1}")

    # Save the final trained model
    torch.save({
        'q_network_state_dict': agent.q_network.state_dict(),
        'target_network_state_dict': agent.target_network.state_dict(),
        'epsilon': agent.epsilon
    }, "drqn_agent_final.pth")

    print("Training completed. Final model saved as 'drqn_agent_final.pth'.")

if __name__ == "__main__":
    main()
