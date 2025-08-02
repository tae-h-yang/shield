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

# Define the neural network for the DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64) 
        self.fc2 = nn.Linear(64, 128)          
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)  # Output layer (one Q-value per action)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

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
#TODO Change gamma to 0.9 so that passing the insertection in the far future makes less preferred.
class DQNAgent:
    def __init__(self, state_size, action_size, actions, batch_size=128, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, learning_rate=0.001):
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

        # Initialize experience replay buffer with more capacity than 2000 as the state size is bigger
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
    # actions = [a for a in np.arange(-4.0, 3.5, 0.5)]  # Available actions for acceleration

    state = None
    past_state = None
    action = None
    reward = None

    state_foe_ids = []

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

            # for i in range(MAX_STATE_FOE_NUMBER):
            #     if i < len(state_foe_ids):
            #         if 
            #         foe_id = foe_ids[i]
            #         state.extend([round(traci.vehicle.getPosition(foe_id)[0], 1),
            #                       round(traci.vehicle.getPosition(foe_id)[1], 1),
            #                       round(traci.vehicle.getSpeed(foe_id), 1),
            #                       round(traci.vehicle.getAcceleration(foe_id), 1),
            #                       traci.vehicle.getImpatience(foe_id)])
            #     else:
            #         state.extend([0, 0, 0, 0, 0])  # Padding

            # print(len(state))
            traci.vehicle.setSpeedMode(ego_vehicle_id, 32)
            action_index = agent.act(state)
            action_value = agent.actions[action_index]  # Map index to actual action
            traci.vehicle.setAcceleration(ego_vehicle_id, action_value, 100)

            reward = -0.5 * step / 10
            max_speed = traci.lane.getMaxSpeed(traci.vehicle.getLaneID(ego_vehicle_id))
            ego_speed = traci.vehicle.getSpeed(ego_vehicle_id)
            if ego_speed > max_speed:
                reward += 0.1 * (max_speed - ego_speed)

            # Penalize stopping at the point far from the stop line
            ego_x = traci.vehicle.getPosition(ego_vehicle_id)[0]
            if ego_speed < 0.01 and (ego_x < 40 or ego_x > 45):
                print("Stopping at the place far from the stop line")
                reward -= 1

            #TODO penalty for emergency braking
            # emergency_stopping_vehicles = traci.simulation.getUniversal(traci.constants.VAR_EMERGENCYSTOPPING_VEHICLES_IDS)
            # if traci.simulation._getUniversal(traci.constants.VAR_EMERGENCYSTOPPING_VEHICLES_NUMBER) > 0:
            #     print("Emergency braking occurred!!")
            #     reward += -5
            # neighbors = traci.vehicle.getNeighbors(ego_vehicle_id, monitoring_radius)

            # for neighbor in neighbors:
            #     foe_id = neighbor[0]

            for foe_id in foe_ids:
                if get_distance(ego_vehicle_id, foe_id) < 16:
                    if traci.vehicle.getAcceleration(foe_id) < -8:
                        reward -= 10

            #TODO Add positive reward for the distance traveled?

            if traci.vehicle.getPosition(ego_vehicle_id)[0] > DISTANCE_THRESH:
                reward += 10
                agent.memory.push((state, action if action is not None else 0, reward, None, 1))
                break

            collision = traci.simulation.getCollidingVehiclesIDList()
            if ego_vehicle_id in collision:
                reward -= 100
                agent.memory.push((state, action if action is not None else 0, reward, None, 1))
                break

            if past_state is not None:  # Ensure valid state
                agent.memory.push((past_state, action if action is not None else 0, reward, state, 0))

            agent.train()

def main():
    actions = [a for a in range(-3, 4)]
    # actions = [-4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 2.6]
    state_size = 5 + MAX_STATE_FOE_NUMBER * 5
    action_size = len(actions)
    agent = DQNAgent(state_size, action_size, actions)

    iterations = 5000
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
