import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
# import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import os
from tensorboardX import SummaryWriter

# Data loading and preprocessing
def mnist_data():
    compose = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((.5, .5, .5), (.5,.5,.5))])
    out_dir = './mnist_data'
    return datasets.MNIST(root = out_dir, train = True,
                          transform = compose, download = True)

# Load data
data = mnist_data()
# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches
num_batches = len(data_loader)

# Defining the Discriminator Network

class Discriminator(torch.nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        n_features = 784
        n_out = 1
        
        self.hidden0 = nn.Sequential(
        nn.Linear(n_features, 1024),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3)
        )
        
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )
        
    def forward(self,x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
    
discriminator = Discriminator()

# Helper functions to convert images to flattened vector and vice-versa
def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

# Defining the Generator Network

class Generator(torch.nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        n_features = 100
        n_out = 784
        
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        
        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
    
generator = Generator()

# Helper function to create random noise
def noise(size):
    n = Variable(torch.randn(size, 100))
    return n

# Defining optimizer
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# Using cross-entropy loss as Discriminator is a binary classifier
loss = nn.BCELoss()

# Defining labels for real and generated images
def ones_target(size):
    data = Variable(torch.ones(size, 1))
    return data

def zeros_target(size):
    data = Variable(torch.zeros(size, 1))
    return data


# Defining training function for generator and discriminator

def train_discriminator(optimizer, real_data, generated_data):
    N = real_data.size(0)
    optimizer.zero_grad() # Reset the gradients
    
    # Train on real Data
    prediction_real = discriminator(real_data)
    error_real = loss(prediction_real, ones_target(N))
    error_real.backward()
    
    # Train on generated data
    prediction_generated = discriminator(generated_data)
    error_generated = loss(prediction_generated, zeros_target(N))
    error_generated.backward()
    
    # Update weights with gradients
    optimizer.step()
    
    total_error = error_real + error_generated
    
    return total_error, prediction_real, prediction_generated
    
def train_generator(optimizer, generated_data):
    N = generated_data.size(0)
    optimizer.zero_grad() # Reset Gradient
    
    # Sample Noise and Generate fake data
    prediction = discriminator(generated_data)
    
    error = loss(prediction, ones_target(N))
    error.backward()
    
    # Update weights with gradients
    optimizer.step()
    
    return error

# Function to save the generated images

def save_images(images, num_images, epoch, n_batch, num_batches, format='NCHW', normalize=True):

	if type(images) == np.ndarray:
		images = torch.from_numpy(images)

	if format == 'NCHW':
		images = images.transpose(1,3)

	step = epoch * num_batches + n_batch

	name_extension = "GAN_MNIST"
	img_name = '{}/images{}'.format(name_extension, '') 

	# Make horizontal grids from image tensor
	horizontal_grid = vutils.make_grid(
		images, normalize=normalize, scale_each=True)

	# Make vertical grid from image tensors
	nrows = int(np.sqrt(num_images))
	grid = vutils.make_grid(
            images, nrow=nrows, normalize=True, scale_each=True)

	writer = SummaryWriter(comment=name_extension)

	# Add horizontal grid to the writer
	writer.add_image(img_name, horizontal_grid, step)

	








