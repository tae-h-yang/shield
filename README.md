# SHIELD (Safe Handling of Intersection Events under other agents' Latent Driving styles)

## Introduction

This project tackles the challenge of an ego vehicle navigating an uncontrolled four-way intersection where other road users have latent driving styles. SHIELD explores solutions using both Markov Decision Processes (MDPs) and Partially Observable Markov Decision Processes (POMDPs) to enhance the safety and efficiency of autonomous vehicles in these complex scenarios.

## Installation

To run this project, you will need to install the SUMO (Simulation of Urban Mobility) software suite.

### macOS Installation

1.  **Update Homebrew**
    ```bash
    brew update
    ```
2.  **Install XQuartz**
    ```bash
    brew install --cask xquartz
    ```
3.  **Tap the SUMO repository**
    ```bash
    brew tap dlr-ts/sumo
    ```
4.  **Install SUMO**
    ```bash
    brew install sumo
    ```

After installation, you may need to log out and back in to allow X11 to start automatically when running SUMO with a GUI. Alternatively, you can start XQuartz manually by pressing `Cmd+Space` and typing "XQuartz".

Finally, make sure to set the `SUMO_HOME` environment variable by adding the following line to your `.zshrc` or `.bashrc` file:

```bash
export SUMO_HOME="/opt/homebrew/opt/sumo/share/sumo"
```

## Running the Simulation

This repository contains several Python scripts for running different simulations and training various models. Below are the commands to execute each of them.

### Random Policy (Baseline)

To run the simulation with a random policy for the ego vehicle, use the following command:

```bash
cd combined_mdp
python random_policy.py 
```

### Combined DQN

To train the Deep Q-Network (DQN) model that considers the ego vehicle and up to 20 other vehicles in a single, high-dimensional MDP, run the following command:

```bash
cd combined_mdp
python dqn.py
```

### Combined DRQN

To train the Deep Recurrent Q-Network (DRQN) model, which is designed to handle partially observable environments where other vehicles' driving styles are unknown, use the following command:

```bash
cd combined_pomdp
python drqn.py
```

### Evaluation

To evaluate a trained model, you can use the corresponding evaluation script. For example, to evaluate the Combined DQN model, run:

```bash
cd combined_mdp
python dqn_eval.py
```

Similarly, to evaluate the Combined DRQN model, you would run:

```bash
cd combined_pomdp
python drqn_eval.py
```