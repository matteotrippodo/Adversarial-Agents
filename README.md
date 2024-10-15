# Competitive Reinforcement Learning Agents
This project implements a competitive reinforcement learning (RL) environment with two agents: one trying to reach a goal and the other acting as an adversary. 
The environment is built from scratch while the agents movements are due to a Deep Q-Network (DQN) with Prioritized Experience Replay (PER), whith supports for both global and independent learning types.

# Dependencies
- Python 3.10+
- PyTorch
- Matplotlib
- Numpy

# Environment
In the first section the class AdversarialEnvorinment is implented, which includes some fundamentals methods for RL envorirments like `step()`, `reset()` and `render()`, also available in the classical OpenAI Gym envs. 

In this implementation the world is a grid, where the agent start form a fixed cell while the goal is generated in a randow cell, far enough for the starting point of the agent.
Through this class we can specify various charateristics of the environment and therefore experiment different settings. In fact, it's possibile to choose:
- height and width of the grid
- the move penalty for the agent
- the reward for reaching the goal and the penalty for the failure
- the number of obstacle within the environment
- the possibility of including an exploration bonus for the agent
- the possibility to have a random moving goal
- the possibility to enable/disable the presence of the enemy
- set a seed to have replicable experiments

When the envoirment is generated, at first, the positions of goal and obstacles are randomly sampled, then the obstaces are kept fixed while the one of the goal varies throughout the episodes.

# Global and Indipendent Learning
In this implementtion two main setting are provided: global and indipendent learing. 

In global learing both the agent and the enemy are moved following a policy based on a unique DQN that models the Q-values for both the agents.
In fact, in a fully competitive setting we can suppose that the Q-values of the agent and its enemy are equal and opposite, since we craft the rewards to be r<sub>1 </sub> = - r<sub>2</sub>. 
Therefore, both the agents can choose their best action based on a minimax policy, meaning that each of them choose the best action given the hypotesis that opponent chooses the action that brings them in the worst scenario.

In case of indipendent learning, each agents has its own DQN and thefore the rewards don't need to related to each other specifically. For this type of learning we need to alternate the training of the two agents: 
while one is training, exploring the action-state space, the other one, in order to be modelled as part of the enviormet, needs to keep its policy fixed. At each step one of the agents does a complete exploration fase starting from a epsilon of 1 to reach ``epsilon_min``.
After multiple steps the agent learns how to reach its goal while avoiding the enemy and therefore failed and ending the episode prematurely.

# Training
To train our agents given a certain enviroment we just need to set some training parameters.

| Parameter         | Description                                           | Default Value   |
|-------------------|-------------------------------------------------------|-----------------|
| `train_episodes`  | Number of episodes for training                       | 5000            |
| `lr`              | Learning rate                                         | 5e-6            |
| `batch_size`      | Size of batch of transition sampled from the buffer   | 64              |
| `gamma`           | Discount factor for future rewards                    | 0.99            |
| `epsilon_decay`   | Rate at which exploration decreases during training   | 0.995           |
| `epsilon_min  `   | Minimum value possible of the exporation factor       | 0.05            |
| `buffer_size  `   | Capacity of the replay buffer used during training    | 100000          |
| `enemy_enabled`   | Boolean flag to enable the adversary agent            | True            |
| `indipendent_learing`| Boolean flag to choose between indipendent and gloabal learing  | False            |

If the environment is generated and the paramenters are all correctly set, the traing is done using the function ``train_agent()``

# Testing
Once the agents are trained the testing is straightforward. In fact, the only important parameter to set is the number of test epsiode. Moreover, thanks to the environment class there is the availablity to create a GIF of the testing episodes.
When the testing episodes end the next plots let us perceive the goodness of the training, showing the total reward for each episode and its time lenght.

After the model is tested is possible to save the learned weights for further future experiments.

