# Using Q-Learning in Frozen Lake Environment

import gym
import random
import numpy as np 
from collections import namedtuple
import collections
import matplotlib.pyplot as plt 


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
'''

# Basic Functions to choose an action following different policies

# Select the action using a epsilon greedy approach 
def select_eps_greedy_action(table, obs, n_actions):
	value, action = best_action_value(table, obs)

	if random.random() < epsilon:
		return random.randint(0, n_actions-1)
	else:
		return action

# Select an action using greedy policy (take the best action according to the policy)
def select_greedy_action(table, obs, n_actions):
	value, action = best_action_value(table,obs)
	return action

# Select the best action to maximise action value function
def best_action_value(table, state):
	best_action = 0
	max_value = 0
	for action in range(n_actions):
		if table[(state,action)]>max_value:
			best_action = action
			max_value = table[(state,action)]

	return max_value, best_action

# Update Q(obs0, action) according to Q(obs1, *) and the reward just obtained
def Q_learning(table, obs0, obs1, reward, action):

	# Take the best value reachable from state obs1
	best_value, _ = best_action_value(table, obs1)
	# Calculate Q-target value
	Q_target = reward + gamma*best_value
	# Calculate Q-error between target and previous value
	Q_error = Q_target - table[(obs0, action)]
	#Update Q(obs0, action)
	table[(obs0, action)] += learning_rate*Q_error

# Test the new table playing test_episodes games
def test_game(env, table, n_actions):
	reward_games = []
	for _ in range(test_episodes):
		obs = env.reset()
		rewards = 0
		while True:
			# env.render() 
			# Render can be uncommented to visualize the path taken y the agent in each game
			next_obs, reward, done, _ = env.step(select_greedy_action(table, obs, n_actions))
			obs = next_obs
			rewards += reward

			if done:
				reward_games.append(rewards)
				break

	return np.mean(reward_games)

# HyperParameters
gamma = 0.95
epsilon = 1
eps_decay_rate = 0.9993
learning_rate = 0.8
test_episodes = 100
max_games = 15001

# Create the Environment
env = gym.make("FrozenLake-v0")
obs = env.reset()

obs_length = env.observation_space.n
n_actions = env.action_space.n

reward_count = 0
games_count = 0

# Create and initialize the table with 0.0
table = collections.defaultdict(float)
    
test_rewards_list = []

while games_count < max_games:

    # Select the action following an ε-greedy policy
    action = select_eps_greedy_action(table, obs, n_actions)
    next_obs, reward, done, _ = env.step(action)

    # Update the Q-table
    Q_learning(table, obs, next_obs, reward, action)

    reward_count += reward
    obs = next_obs

    if done:
        epsilon *= eps_decay_rate

        # Test the new table every 1000 games
        if games_count % 1000 == 0:
            test_reward = test_game(env, table, n_actions)
            print('Ep:', games_count, 'Test reward:', test_reward, np.round(epsilon,2))

            test_rewards_list.append(test_reward)

        obs = env.reset()
        reward_count = 0
        games_count += 1    

# Plot the accuracy over the number of steps
# print("Final Q value table", table)

plt.figure(figsize=(10,10))
plt.xlabel('Steps')
plt.ylabel('Accurracy')
plt.plot(test_rewards_list)
plt.show()
