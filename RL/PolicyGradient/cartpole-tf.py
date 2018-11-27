import tensorflow as tf 
import gym
import numpy as np 
import matplotlib.pyplot as plt 

# Create the environment
env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(1)

# Defining hyperparameters
state_size = 4
action_size = env.action_space.n

max_episodes = 300
learning_rate = 0.01
gamma = 0.99

# To calculate the cumulative discounted reward 
def discount_and_normalize_rewards(episode_rewards):

	discounted_episode_rewards = np.zeros_like(episode_rewards)
	cumulative = 0.0

	for i in reversed(range(len(episode_rewards))):
		cumulative = cumulative*gamma + episode_rewards[i]
		discounted_episode_rewards[i] = cumulative

	mean = np.mean(discounted_episode_rewards)
	std = np.std(discounted_episode_rewards)
	discounted_episode_rewards = (discounted_episode_rewards - mean)/std

	return discounted_episode_rewards

# Defining our policy gradient neural network 
# The network takes into input the current state og the environment
# It outputs a probability distribution in action space via the softmax function
# 

with tf.name_scope("policy_gradient"):

	input_ = tf.placeholder(tf.float32, [None, state_size], name = "input_")
	actions = tf.placeholder(tf.float32, [None, action_size], name = "actions")
	discounted_episode_rewards_ = tf.placeholder(tf.float32, [None,], name = "discounted_episode_rewards_")

	mean_reward_ = tf.placeholder(tf.float32, name = "mean_reward_")

	with tf.name_scope("layer1"):
		layer1 = tf.contrib.layers.fully_connected(
			inputs = input_, 
			num_outputs = 10, 
			activation_fn = tf.nn.relu, 
			weights_initializer=tf.contrib.layers.xavier_initializer()
			)

	with tf.name_scope("layer2"):
		layer2 = tf.contrib.layers.fully_connected(
			inputs = layer1, 
			num_outputs = action_size,
			activation_fn = tf.nn.relu, 
			weights_initializer = tf.contrib.layers.xavier_initializer()
			)

	with tf.name_scope("layer3"):
		layer3 = tf.contrib.layers.fully_connected(
			inputs = layer2, 
			num_outputs = action_size,
			activation_fn = None,
			weights_initializer = tf.contrib.layers.xavier_initializer()
			)

	with tf.name_scope("softmax"):
		action_distribution = tf.nn.softmax(layer3)

	with tf.name_scope("loss"):
		# tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
		entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = layer3, labels = actions)
		loss = tf.reduce_mean(entropy*discounted_episode_rewards_)

	with tf.name_scope("train"):
		train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# Training the agent

all_rewards = []
total_rewards = []
total_score = 0
maximum_rewards_recorded = 0
episode = 0
running_reward = 10
running_reward_list = []

episode_list = []
reward_list = []

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for episode in range(max_episodes):

		episode_states, episode_actions, episode_rewards = [], [], []
		episode_rewards_sum = 0

		state = env.reset()

		# env.render()

		for t in range(500):

			action_probability_distribution = sess.run(
				action_distribution, 
				feed_dict = {input_ : state.reshape([1,4])})

			action = np.random.choice(range(action_probability_distribution.shape[1]), 
				p=action_probability_distribution.ravel()) 

			new_state, reward, done, info = env.step(action)

			episode_states.append(state)

			action_ = np.zeros(action_size)
			action_[action] = 1

			episode_actions.append(action_)
			episode_rewards.append(reward)

			if done:
				episode_rewards_sum = np.sum(episode_rewards)
				all_rewards.append(episode_rewards_sum)

				total_rewards = np.sum(all_rewards)
				mean_reward = np.divide(total_rewards, episode+1)

				maximum_rewards_recorded = np.amax(all_rewards)

				running_reward = 0.99*running_reward + 0.01*episode_rewards_sum

				print("-----------------------------------------")
				print("Episode: ", episode)
				print("Reward: ", episode_rewards_sum)
				print("Maximum reward so far: ", maximum_rewards_recorded)

				if episode %10 ==0:
					episode_list.append(episode)
					reward_list.append(episode_rewards_sum)


				discounted_episode_rewards = discount_and_normalize_rewards(episode_rewards)

				loss_, _ = sess.run([loss, train_opt], 
					feed_dict={input_: np.vstack(np.array(episode_states)),
					actions: np.vstack(np.array(episode_actions)),
					discounted_episode_rewards_: discounted_episode_rewards 
					})
                
				episode_states, episode_actions, episode_rewards = [],[],[]

				break

			state = new_state

		running_reward_list.append(running_reward)
		if running_reward > env.spec.reward_threshold:
			print("Agent is trained!!")
			break

	print("Running Reward: ", running_reward)
	print("Calculating average score: \n\n")

	for i in range(100):
		state = env.reset()
		done = False
		episode_score = 0

		while not done:
			action_probability_distribution = sess.run(
				action_distribution, 
				feed_dict = {input_ : state.reshape([1,4])})

			action = np.argmax(action_probability_distribution)

			state, reward, done, info = env.step(action)
			episode_score += reward

			if done:
				break

		print(episode_score)
		total_score += episode_score


print(total_score/100)

plt.plot(episode_list, reward_list)

# plt.plot(running_reward_list)
plt.title("Episode Reward Variation")
plt.show()




		







