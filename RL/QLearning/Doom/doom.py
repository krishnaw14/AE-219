# Implementation of an agent that plays doom using Deep Q Learning

import tensorflow as tf 
import numpy as np 
from vizdoom import *

import random
import time
from skimage import transform

from collections import deque
import matplotlib.pyplot as plt 
import warnings

warnings.filterwarnings('ignore')

'''
The environment has following characteristics: 
- A monster is spawned randomly somewhere along the opposite wall
- Agent can only go left or right or shoot
- Monster is killed by one hit
- Episode finishes when monster is killed or on timeout

This environment takes:

A configuration file to handle all the options such as size of the frame, possible actions,etc.
A scenario file that generates the correct scenario

This model is trained for 3 possible actions [[0,0,1], [1,0,0], [0,1,0]] so one hot encoding is not required

Rewards are defined as follows:
- 101 for killing the monster
- -5 for missing
- -1 as living reward 
'''

# Create the environment
def create_environment():
	game = DoomGame()

	# Load the current configuration
	game.load_config("basic.cfg")

	# Load the correct scenario (basic scenario in this case)
	game.set_doom_scenario_path("basic.wad")

	game.init()

	left = [1,0,0]
	right = [0,1,0]
	shoot = [0,0,1]
	possible_actions = [left, right, shoot]

	return game, possible_actions

# Random action to test the environment
def test_environment():
	game = DoomGame







