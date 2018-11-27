import gym
import math
import random
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.transforms as T 

# Create the environment
env = gym.make('CartPole-v0').unwrapped

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creating Replay Buffer 
'''
A Replay Buffer stores the transition that the agent obsereves, this allows us to reuse this data.
By sampling from it randomly, the transitions that build up a batch are decorrelated. 
This greatly stabilizes and improves the DQN training procedure.
'''

# Transition is a named tuple representing a single transition in pour environment
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# ReplayMemory is a cyclic buffer of bounded size that holds the transitions observed recently.
class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args): # To save a transition
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size): # Selecting a random batch for training
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)

# DQ Network - takes as input the difference between the current and precious screen patches
# Output has 2 dimensions for right and left
# Network is trying to predict the Q value of taking each action given the current state

class DQN(nn.Module):

	def __init__(self):

		super(DQN, self).__init__()

		self.layer1 = nn.Sequential(nn.Conv2d(3,16,kernel_size = 5, stride = 2), 
			nn.BatchNorm2d(16))
		self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=5, stride=2), 
			nn.BatchNorm2d(32))
		self.layer3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=5, stride=2),
			nn.BatchNorm2d(32))
		self.head = nn.Linear(448, 2)

	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		x = F.relu(self.layer3(x))
		x = self.head(x.view(x.size(0), -1))

# Extracting and processing renderd images from the environment

resize = T.compose([T.ToPILImage(), 
	T.ReSize(40, interpolation = Image.CUBIC), 
	T.ToTensor()])

screen_width = 600 # For gym env

def get_cart_location():
	world_width = env.x_threshold * 2
	scale = screen_width / world_width
	return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)
    # Strip off the top and bottom of the screen
	screen = screen[:, 160:320]
	view_width = 320
	cart_location = get_cart_location()
	if cart_location < view_width // 2:
		slice_range = slice(view_width)
	elif cart_location > (screen_width - view_width // 2):
		slice_range = slice(-view_width, None)
	else:
		slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
	screen = screen[:, :, slice_range]
    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
	screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
	screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
	return resize(screen).unsqueeze(0).to(device)


env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()








