import gym
import numpy as np 

env = gym.make('CartPole-v1')

def play(env, policy, train):
	observation = env.reset() # Resets the game to starting state

	done = False # To track if the game is over yet (done = True when game is over)
	score = 0 # Total score of the policy
	observations = [] # snapshot of each step during the training

	# Play the game for large number of timesteps until gym tells us than the game is done
	for i in range(500):

		observations = [observation.tolist()]

		# If simulation was over the last iteration, exit the loop
		if done: 
			# print("-------Game Over-----------")
			# print("Score = ", score)
			break

		'''
		We will pick an action according to the policy matrix
		Observation is the state of the agent that is essentially a 1D array with 4 elements,
		corresponding to the position of the cart, speed of the cart, angular position of the pole and angular velocity of the pole.
		Policy is a mapping from the state of the agent to the next action of the agent. It is 
		Here, the outcome is simply the dot product of observation and policy. If this dot product > 0, action = 1 and the car moves right.
		'''
		outcome = np.dot(policy, observation)
		action = 1 if outcome > 0 else 0

		# Visualize the simulation
		env.render()

		# Make the action and store the reward
		observation, reward, done, info = env.step(action)
		score += reward

		# Printing the Score and iteration for each policy to observe with the rendered simulation 
		if i%10 == 0 and train == True:	
			print("Iteration = ", i, "Score = ", score, "Information = ", info)

	return score, observations


best_parameters = (0, [], []) # To store the best score, observation and best performing policy

# Policy is a 1D array of 4 elements. Dot product of policy is taken with the current state of the agent to get the required action.

# Training on various prandomly generated policies
for i in range(10):
	print("\n--------------------------------\n")
	print("Trying out", i+1, "th policy")
	# policy = np.random.rand(1,4) - 0.5
	policy = np.random.rand(1,4) - 0.5

	score, observations = play(env, policy, True)
  
	if score > best_parameters [0]: 
		best_parameters = (score, observations, policy)

print("\n--------------------------------\n")
print("Training Over!")
print('Maximum Score', best_parameters [0])
print('Maximum Observation', best_parameters [1])
print('Best Policy', best_parameters [2])
print("\n--------------------------------\n")
print("Evaluating Average Score of the best performing Policy")

total_score = 0
for _ in range(100):
  score, _  = play(env, best_parameters[2], False)
  total_score += score
  
print('Average Score (100 trials)', total_score/100)


