import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

import datetime
import numpy as np
import scipy.misc as misc

from helper_function import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_cuda(x):
  return x.to(device)

class Discriminator(nn.Module):

#'''
#Discriminator Neural Network Class
#Trained with BCEloss to output 1 for real images and 0 for test images 

#Input = (batch_size, 28, 28, 1)
#layer_1_output = (batch_size, 14, 14, 64)
#layer_2_output = (batch_size, 7, 7, 128) --> reshaped as (batch_size, 128*7*7)
#layer_3_output = (batch_size, 1024) 
#layer_4_output = (batch_size, 1)

#layer_3 and layer_4 output are returned. 
#layer_3_output is fed as input to the Q network. 
#'''

  def __init__(self):
    
    super(Discriminator, self).__init__()
    
    self.layer_1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size = 3, stride = 2, padding = 1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2),
    )
    
    self.layer_2 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size = 3, stride = 2 , padding = 1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2)
    )
    
    self.layer_3 = nn.Sequential(
    	nn.Linear(128*7*7 , 1024),
    	nn.BatchNorm1d(1024), 
    	nn.LeakyReLU(0.2),
    	)
    
    self.layer_4 = nn.Sequential(
    	nn.Linear(1024, 1),
    	nn.Sigmoid(),
    	)
    
  def forward(self , x):
    
    y = self.layer_1(x)
    y = self.layer_2(y)
    y = y.view(y.size(0), -1)
    y = self.layer_3(y) # Upto Layer 3 is same for the Q Network as well
    D_out = self.layer_4(y)
    
    return D_out, y

class RecognitionNetwork(nn.Module): # Or the Q-Network
  
#'''
#Recognition Network Q is same as Discriminator till layer 3. 

#Input = (batch_size, 1024)
#layer_4_output = (batch_size, 14)

#The layer_4_output is split into y_discrete, y_mu and y_var.

#y_discrete - For the digit labels 0 to 9
#y_mu - mean of the codes that are supposed to capture rotation and thickness
#y_var - variance of the codes that are supposed to capture rotation and thickness
	
#'''
  
  def __init__(self):
    super(RecognitionNetwork, self).__init__()
    
    self.layer_4 = nn.Sequential(nn.Linear(1024, 128), 
                                nn.BatchNorm1d(128),
                                nn.LeakyReLU(0.2),
                                nn.Linear(128, 14)
                                )
    
  def forward(self, y):
    y = self.layer_4(y)
    
    y_discrete = F.softmax(y[:, 0:10])
    y_mu = y[:, 10:12]
    y_var = y[:, 12:14].exp()
  
    return y_discrete, y_mu, y_var

class Generator(nn.Module):

#'''
#Generator Network
#Trained with BCEloss for the discriminator to output 1 on the generated images

#Input = (batch_size, 74)
#layer_1_output = (batch_size, 1024, 1)
#layer_2_output = (batch_size, 128*7*7) --> reshaped as (batch_size, 128, 7, 7)
#layer_3_output = (batch_size, 64, 14, 14)
#layer_4_output = (batch_size, 1, 28, 28)

#'''
  
  def __init__(self, z_dim = 62, code_dim = 12, num_classes = 784):
    
    super(Generator, self).__init__()
    
    input_dim = z_dim + code_dim
    
    self.layer_1 = nn.Sequential(
        nn.Linear(input_dim, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
    )
    
    self.layer_2 = nn.Sequential(
        nn.Linear(1024, 128*7*7),
        nn.BatchNorm1d(128*7*7),
        nn.ReLU(),
    )
    
    self.layer_3 = nn.Sequential(
        nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
    )
    
    self.layer_4 = nn.Sequential(
        nn.ConvTranspose2d(64, 1, kernel_size = 4, stride = 2 , padding = 1),
        nn.Tanh(),
    )
    
  def forward(self, z, code):
    z = z.view(z.size(0), -1)
    code = code.view(code.size(0), -1)
    x = torch.cat((z,code), 1)
    y = self.layer_1(x)
    y = self.layer_2(y)
    y = y.view(y.size(0), 128, 7, 7)
    y = self.layer_3(y)
    G_out = self.layer_4(y)
    
    return G_out

# Defining the transformation and dataloader for mnist dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5, 0.5, 0.5)),
    ])

batch_size = 128

mnist = datasets.MNIST('./data/mnist/', train = True, 
                       transform = transform, download = True)

data_loader = DataLoader(mnist, batch_size = batch_size, shuffle = True, drop_last = True)

D = Discriminator()
G = Generator()
Q = RecognitionNetwork()

D = to_cuda(D)
G = to_cuda(G)
Q = to_cuda(Q)

BCE_loss = nn.BCELoss()
CE_loss = nn.CrossEntropyLoss()

D_opt = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.99))
G_opt = torch.optim.Adam([{'params':G.parameters()}, {'params':Q.parameters()}], 
                         lr=1e-3, betas=(0.5, 0.99))


max_epochs = 201
step = 0

z_dim = 62
discrete_dim = 10
continuous_dim = 2

D_labels = to_cuda(torch.ones(batch_size))
D_fakes = to_cuda(torch.zeros(batch_size))

for epoch in range(max_epochs):
  for idx, (images, labels) in enumerate(data_loader):
    
    step += 1
    # batch_size = images.size(0)
    labels = labels.view(batch_size, 1)
    
    x = to_cuda(images)
    
    # Training Discriminator
    D_out , D_layer3_out = D(x)
#     print(D_out.shape)
#     print(D_labels.shape)
    D_loss_real = BCE_loss(D_out, D_labels)
    
    z, code = sample_noise(batch_size, z_dim, discrete_dim, continuous_dim,
                          label = labels, supervised = True)
    
    D_gen, D_gen_layer3_out = D(G(z, code))
    D_loss_fake = BCE_loss(D_gen, D_fakes)
    
    D_loss = D_loss_fake + D_loss_real
    
    D_opt.zero_grad()
    D_loss.backward()
    D_opt.step()
    
    if step > 400:
      # Training Generator
      z, code = sample_noise(batch_size, z_dim, discrete_dim, continuous_dim,
                            label = labels, supervised = True)
      
      discrete_label = torch.max(code[:, :-2], 1)[1].view(-1, 1)
      
      G_out, features = D(G(z, code))
      y_discrete, y_mu, y_var = Q(features)
      
      G_loss = BCE_loss(G_out, D_labels)
      Q_loss_discrete = CE_loss(y_discrete, discrete_label.view(-1))
      Q_loss_continuous = -torch.mean(torch.sum(log_gaussian(
          code[:, -2:], y_mu, y_var), 1))
      mutual_info_loss = Q_loss_discrete + Q_loss_continuous*0.1
      GnQ_loss = G_loss + mutual_info_loss
      
      G_opt.zero_grad()
      GnQ_loss.backward()
      G_opt.step()
      
      
    if step%1000 == 0:
      print('Epoch: {}/{}, Step: {}, D Loss: {}, G Loss: {}, GnQ Loss: {}, Time: {}'
            .format(epoch, max_epochs, step, D_loss.data[0], G_loss.data[0], 
                    GnQ_loss.data[0], str(datetime.datetime.today())[:-7]))
      
  if epoch % 5 == 0:
    G.eval()
    img1, img2, img3 = get_sample_image(z_dim, continuous_dim, G)
    misc.imsave('Results/InfoGAN_epoch_{}_type1.png'.format(epoch), img1)
    misc.imsave('Results/InfoGAN_epoch_{}_type2.png'.format(epoch), img2)
    misc.imsave('Results/InfoGAN_epoch_{}_type3.png'.format(epoch), img3)
      
    G.train()
      

