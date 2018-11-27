import tensorflow as tf 
import numpy as np 
import retro

from skimage import transform
from skimage.color import rgb2gray

import matplotlib.pyplot as plt
from collections import deque

import random
import warnings
warnings.filterwarnings('ignore')

'''
Define the environment
'''

env = retro.make(game = 'SpaceInvaders-Atari2600')

'''
For this environment: 
Frame size = (210,160,3)
action size = 8
'''

# Create one hot encoded version of the actions
possible_actions = np.array(np.identity(env.action_space.n, dtype = int).tolist())

'''
Defining the preprocessing function applied to very frame before it is fed to the neural network

The frame is converted to grayscale image and then resized and normalized
'''

def preprocess_frame(frame):

	gray = rgb2gray(gray)
	cropped_frame = gray[8:-12. 4:-12]
	normalized_frame = cropped_frame/255.0
	preprocessed_frame = transform.resize(normalized_frame, [110,84])

	return preprocessed_frame

'''
Defining a function to stack the frames
Stacking frames gives a sense of motion to our Neural Network
Not every frame is stacked. Every fourth frame is considered and stacked\

Frame skipping is already implemented in the library

First the frames are preprocessed and then appended to deque that automatically removes the oldest frames.
Then finally stacked state is built
So for the first stacked frame, we need 4 frames and then at each timestep, a new frame is added to the deque and stacked to form a new stacked frame

'''

stack_size = 4

# Initialize deque with zero-images one array for each image
stacked_frames = deque([np.zeros((110,84), dtype = np.int) for i in range(stack_size)],
 maxlen = 4)

def stack_frames(stacked_frames, state, is_new_episode):

	frame = preprocess_frame(state)

	if is_new_episode:
		# Clear the stacked frames for new episode
		stacked_frames = deque([np.zeros((110,84), dtype = np.int) for i in range(stack_size)],
			maxlen = 4) 

		stacked_frames.append(frame)
		stacked_frames.append(frame)
		stacked_frames.append(frame)
		stacked_frames.append(frame)

		stacked_state = np.stack(stacked_frames, axis = 2)

	else:

		stacked_frames.append(frame)

		stacked_state = np.stack(stacked_frames, axis = 2)

	return stacked_state, stacked_frames

# Model Hyperparameters
state_size = [110,84,4] # 4 frames of size (110,84) stacked
action_size = env.action_space.n # 8 possible actions in this case
learning_rate = 0.00025

# Training Hyperparameters
total_episodes = 50
max_steps = 50000
batch_size = 64

# Exploration-Exploitation tradeoff parameters
explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.00001

# Q Learning parameters
gamma = 0.9

# Memory HyperParameters
pretrain_length = batch_size
memory_size = 1000000

# Preprocessing hyperparameters
stack_size = 4

training = False
episode_render = False

# Creating the Deep Q-Learning Neural Network model

class DQNetwork:

	def __init__(self, state_size, action_size, learning_rate, name = 'DQNetwork'):
		self.state_size = state_size
		self.action_size = action_size
		self.learning_rate = learning_rate

		with tf.variable_scope(name):
			self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name = "inputs_")
			self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")

			self.target_Q = tf.placeholder(tf.float32, [None], name = "target")

			self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
				filters = 32, kernel_size = [8,8], strides = [4,4],
				padding = "VALID",
				kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
				name = "conv1")

			self.conv1_out = tf.nn.elu(self.conv1, name = "conv1_out")

			self.conv2 = tf.layers.conv2d(inputs = self.conv1_out, filters = 64,
				kernel_size = [4,4], strides = [2,2], padding = "VALID", 
				kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
				name = "conv2")

			self.conv2_out = tf.nn.elu(self.conv2, name = "conv2_out")

			self.conv3 = tf.layers.conv2d(inputs = self.conv2_out, filters = 64,
				kernel_size = [3,3], strides = [2,2], padding = "VALID", 
				kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
				name = "conv3")

			self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

			self.flatten = tf.contrib.layers.flatten(self.conv3_out)

			self.fc = tf.layers.dense(inputs = self.flatten, units = 512, 
				activation = tf.nn.elu, 
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
				name = "fc")

			self.output = tf.layers.dense(inputs = self.fc, 
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
				units = self.action_size, activation = None)

			self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))

			self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

			self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

tf.reset_default_graph()

DQNetwork = DQNetwork(state_size, action_size, learning_rate)

# Experience Replay

class Memory():
	def __init__(self, max_size):
		self.buffer = deque(maxlen = max_size)
    
	def add(self, experience):
		self.buffer.append(experience)
    
	def sample(self, batch_size):
		buffer_size = len(self.buffer)
		index = np.random.choice(np.arange(buffer_size),
			size = batch_size,replace = False)
        
		return [self.buffer[i] for i in index]

memory = Memory(max_size = memory_size)
for i in range(pretrain_length):
    # If it's the first step
	if i == 0:
		state = env.reset()
        
		state, stacked_frames = stack_frames(stacked_frames, state, True)
        
    # Get the next_state, the rewards, done by taking a random action
    choice = random.randint(1,len(possible_actions))-1
    action = possible_actions[choice]
    next_state, reward, done, _ = env.step(action)
    
    #env.render()
    
    # Stack the frames
	next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
    
    
    # If the episode is finished (we're dead 3x)
	if done:
        # We finished the episode
		next_state = np.zeros(state.shape)
        
        # Add experience to memory
		memory.add((state, action, reward, next_state, done))
        
        # Start a new episode
		state = env.reset()
        
        # Stack the frames
		state, stacked_frames = stack_frames(stacked_frames, state, True)
        
	else:
        # Add experience to memory
		memory.add((state, action, reward, next_state, done))
        
        # Our new state is now the next_state
		state = next_state

# Setup TensorBoard Writer
writer = tf.summary.FileWriter("/tensorboard/dqn/1")

## Losses
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()















