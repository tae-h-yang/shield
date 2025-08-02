import traci
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Constants
MAX_STATE_FOE_NUMBER = 10
DISTANCE_THRESH = 60  # Distance threshold for ego vehicle to pass intersection

# Define the neural network for the DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)          # Second fully connected layer
        self.fc3 = nn.Linear(64, action_size)  # Output layer (one Q-value per action)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# DQN Agent with Training Loop
class DQNAgent:
    def __init__(self, state_size, action_size, actions, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.actions = actions  # Store the actions list
        self.batch_size = batch_size
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # How fast to decay epsilon
        self.epsilon_min = epsilon_min  # Minimum epsilon value
        self.learning_rate = learning_rate  # Learning rate for the optimizer

        # Create the Q-network
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Copy the weights of the Q-network to the target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Initialize experience replay buffer
        self.memory = ReplayBuffer(10000)

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randint(0, len(self.actions) - 1)  # Return a random index
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_network(state)  # Get Q-values from the Q-network
            return torch.argmax(q_values).item()  # Return the index of the best action

    def train(self):
        # Only train if we have enough samples in memory
        if self.memory.size() < self.batch_size:
            return

        # Sample a batch of experiences from the replay buffer
        batch = self.memory.sample(self.batch_size)

        # Unpack the batch
        states, actions, rewards, next_states, dones = zip(*batch)

        # Ensure consistent shapes by padding
        def pad_state(state, size):
            if state is None:
                return [0] * size  # Replace None with a zero-filled state
            return state + [0] * (size - len(state)) if len(state) < size else state

        states = [pad_state(s, self.state_size) for s in states]
        next_states = [pad_state(ns, self.state_size) for ns in next_states]

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Get the Q-values for the current states from the Q-network
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get the Q-values for the next states from the target network
        next_q_values = self.target_network(next_states)
        next_q_values = next_q_values.max(1)[0]

        # Compute the target Q-values
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute the loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon (decay)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        # Periodically copy the Q-network weights to the target network
        self.target_network.load_state_dict(self.q_network.state_dict())

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
    # actions = [a for a in np.arange(-4.0, 3.5, 0.5)]  # Available actions for acceleration

    state = None
    past_state = None
    action = None
    reward = None

    while True:
        traci.simulationStep()
        step += 1
        ego_vehicle_id = "ego.0"

        if ego_vehicle_id in traci.vehicle.getIDList():
            past_state = state
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

            # print(len(state))
            traci.vehicle.setSpeedMode(ego_vehicle_id, 32)
            action_index = agent.act(state)
            action_value = agent.actions[action_index]  # Map index to actual action
            traci.vehicle.setAcceleration(ego_vehicle_id, action_value, 100)

            reward = -0.1 * step / 10
            max_speed = traci.lane.getMaxSpeed(traci.vehicle.getLaneID(ego_vehicle_id))
            ego_speed = traci.vehicle.getSpeed(ego_vehicle_id)
            if ego_speed > max_speed:
                reward += 0.1 * (max_speed - ego_speed)

            #TODO penalty for emergency braking

            if past_state is not None:  # Ensure valid state
                agent.memory.push((past_state, action if action is not None else 0, reward, state, 0))

            if traci.vehicle.getPosition(ego_vehicle_id)[0] > DISTANCE_THRESH:
                reward += 10
                agent.memory.push((state, action if action is not None else 0, reward, None, 1))
                break

            collision = traci.simulation.getCollidingVehiclesIDList()
            if ego_vehicle_id in collision:
                reward -= 100
                agent.memory.push((state, action if action is not None else 0, reward, None, 1))
                break

            agent.train()

            # if step % 100 == 0:
            #     agent.update_target_network()

    # print('reward: ', reward)

# def main():
#     # actions = [a for a in np.arange(-4.0, 3.5, 0.5)]
#     actions = [a for a in range(-4, 4)]
#     state_size = 5 + MAX_STATE_FOE_NUMBER * 5
#     action_size = len(actions)
#     agent = DQNAgent(state_size, action_size, actions)

#     iterations = 20
#     for episode in range(iterations):
#         start_simulation()
#         run_simulation(agent)
#         close_simulation()

#         if episode % 10 == 0:
#             agent.update_target_network()

#         print(f"Episode {episode + 1}/{iterations} completed")

#     start_simulation("sumo-gui")
#     run_simulation(agent)
#     close_simulation()

def main():
    actions = [a for a in range(-4, 4)]
    # actions = [-4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 2.6]
    state_size = 5 + MAX_STATE_FOE_NUMBER * 5
    action_size = len(actions)
    agent = DQNAgent(state_size, action_size, actions)

    iterations = 20
    for episode in range(iterations):
        start_simulation()
        run_simulation(agent)
        close_simulation()

        if episode % 10 == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}/{iterations} completed")

    # Save the trained model
    torch.save({
        'q_network_state_dict': agent.q_network.state_dict(),
        'target_network_state_dict': agent.target_network.state_dict(),
        'epsilon': agent.epsilon
    }, "dqn_agent.pth")

    print("Training completed. Model saved to 'dqn_agent.pth'.")

    # # Optional: Run simulation with GUI for visualization
    # start_simulation("sumo-gui")
    # run_simulation(agent)
    # close_simulation()


if __name__ == "__main__":
    main()
