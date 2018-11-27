# Using Q-Learning in FrozenLake Environment

'''
FrozenLake-v0 Environment

The agent controls the movement of a character in a grid world. Some tiles of the grid are walkable, 
and others lead to the agent falling into the water. Additionally, the movement direction of the agent is uncertain 
and only partially depends on the chosen direction. The agent is rewarded for finding a walkable path to a goal tile.

The surface is described using a grid like the following:

SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)

The episode ends when you reach the goal or fall in a hole. You receive a reward of 1 if you reach the goal, and zero otherwise.

More details can be found here - https://gym.openai.com/envs/FrozenLake-v0/

In a nutshell, the goal of the game is to go from start state (S) to goal state (G) by walking just on the frozen states (F) and avoid holes (H). 
'''

import numpy as np 
import gym
import random

# Creating the environment
env = gym.make("FrozenLake-v0")

# Create the Q table

'''
Q-table comprises maximum expected reward for each action at each stage. 
The rows of the table correspond to the states while the columns correspond to the actions that the agent can take under the given policy.
Each value in the Q-table (Q-table score) is the maximum expected future reward that I'll get if I take that action at that stage with 
the given best policy.
'''

action_size = env.action_space.n # Possible number of states (= 16 for frozenLake environment)
state_size = env.observation_space.n # Possible number of actions (= 4 for frozenLake environment)

Qtable = np.zeros((state_size, action_size))

# HyperParameters
total_episodes = 15000
learning_rate = 0.8
max_steps = 99 # Maximum steps that the agent can take per episode
gamma = 0.95 # Discount rate

# Exploration parameters
epsilon = 1.0 # Exploration rate at the start - is high as initially we want our agent to explore the environment as much as possible
max_epsilon = 1.0
min_epsilon = 0.01 # Minimum exploration probability 
decay_rate = 0.005 # Exponential decay rate for exploration rate

# Implementing the Q learning algorithm

rewards = [] # List of rewards

for episode in range(total_episodes):

	# Reset the environment and parameters before each episode
	state = env.reset() 
	step = 0
	done = False
	total_reward = 0

	for step in range(max_steps):
		tradeoff = random.uniform(0,1) # Represents the tradeoff between exploration and exploitation

		if tradeoff > epsilon: # We exploit the environment in this case chosing the best action for that state (greedy approach)
			action = np.argmax(Qtable[state,:])

		else: # We explore the environment in this case chosing any random action
			action = env.action_space.sample()

		# Take the action and obtain the reward and new state for this action
		new_state, reward, done, info = env.step(action)

		# Update Q table using Bellman Ford equation
		Qtable[state,action] = Qtable[state,action] + learning_rate*(reward + gamma*np.max(Qtable[new_state, :]) - Qtable[state,action] )

		# Update the state and reward
		total_reward = total_reward + reward
		state = new_state

		# If the episode is ended
		if done == True:
			break

	# Update epsilon as we aim to increase exploitation and reduce exploration as the training proceeds because the Q table starts to take shape after every episode iteration.
	epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
	rewards.append(total_reward)


print("Average score over training:", np.mean(np.array(rewards))) 
print("Final Q-table after training:", Qtable)


# Testing the algorithm
print("Testing the algorithm on 5 episodes...")

env.reset()
for episode in range(5):
	state = env.reset()
	step = 0
	done = False
	print("---------- Episode",episode, "---------")

	for step in range(max_steps):
		action = np.argmax(Qtable[state,:])
		new_state, reward, done, info = env.step(action)

		state = new_state

		if done:
			env.render() # To verify that our agent reaches the goal state
			print("Number of steps = ", step)
			break

env.close()














