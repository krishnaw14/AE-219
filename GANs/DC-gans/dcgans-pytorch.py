import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

import numpy as np
import scipy.misc

'''
is_available() function checks for tha availability of CUDA enabled GPU for parallelization.
Device represents the device on which a torch.Tensor is allotted.
A torch.Tensor's device can be accessed via Tensor.device property.
to() function in pytorch performs conversion of tensor datatype or device conversion.
'''

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
    
def to_cuda(x):
    return x.to(device)

'''
one_hot_encoder() function computes one hot encoding of the output label of the digit
torch.LongTensor() creates a tensor of the specified shape
zero_() fills the tensor with zeros
scatter_(dim,index,src) function writtes the values from src (1) to self(one_hot) 
at the index specified by the index tensor (y). 
'''
def one_hot_encoder(y, num_classes=10):
    one_hot = torch.LongTensor(y.size(0), num_classes).zero_()
    one_hot.scatter_(1,y,1)
    return one_hot

'''
Printing the out images from the generator
Print 100 output images in 10x10 grid
torch.cat() concatenates the given sequence of tensros in the given dimension. 
For 10x10 grid of output images, 10 images are concatenated in vertical dimension (stored in line_img)
and all the vertical line of images are concatenated horizontally (stored in all_img).
'''
def get_output_image(latent_dim=100):
    for num in range(10):
        for i in range(10):
            z = to_cuda(torch.randn(1, latent_dim))
            out = G(z)
            line_img = torch.cat((line_img, out.view(28, 28)), dim=1) if i > 0 else out.view(28, 28)
        all_img = torch.cat((all_img, line_img), dim=0) if num > 0 else line_img
    img = all_img.cpu().data.numpy()
    return img

'''
Discriminator architecture as per the paper.
The class inherits from nn.Module whose forward function is overridden as per the model.
The model is defined in the constructor and the forward propagation is described in forward()
First three layers are a bloack consisting of Convolution, Batch Normalization and LeakyRelu functions.
Last layer is a fully connected layer with sigmoid activation. 

Discriminator input = image of size (28,28)
Discriminator output = 0 or 1
'''
class Discriminator(nn.Module):

    def __init__(self, in_channels = 1, num_classes = 1):
        
        super(Discriminator, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(256,512, 3, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(4)
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(512,1),
            nn.Sigmoid()
        )
        
    def forward(self, x, y = None):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        
        return out
'''
Generator architecture as per the paper.
The class inherits from nn.Module whose forward function is overridden as per the model.
The model is defined in the constructor and the forward propagation is described in forward()
Last three layers are a bloack consisting of Convolution, Batch Normalization and ReLu functions.
First layer is a fully connected layer with ReLU activation. 

Generator input: Random noise with input_size = 100
Generator output: Vector of size (784,1) which is reshaped as (28,28) for image generation
'''

class Generator(nn.Module):

    def __init__(self, input_size=100, label_size=10, num_classes=784):
        super(Generator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 4*4*512),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        
    def forward(self, x, y=None):
        x = x.view(x.size(0), -1)
        out = self.layer1(x)
        out = out.view(out.size(0), 512, 4, 4)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

# Creating object of discriminator and generator 
D = to_cuda(Discriminator())
G = to_cuda(Generator())

# Dataloading and proprocessing
'''
transforms.Compose() composes several transforms of the data together
transforms.ToTensor() converts PIL image to numpy array
transforms.Normalize() normailizes the images with the mean and std along each channel
MNIST dataset loader is predefined in pytorch
'''
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                std=(0.5, 0.5, 0.5))]
)

mnist = datasets.MNIST(root = 'MNIST-data/', train = True,
                       transform = transform, download = False)

batch_size = 128

data_loader = DataLoader(dataset=mnist, batch_size = batch_size,
                         shuffle = True, drop_last = True)
'''
Binary Cross entropy loss is defined for training the Discriminator and generator with Adam optimizer.
Discriminator is trained to output 1 for real images and 0 for generator output images.
Generator is trained to output images such that discriminator outputs 1 for those images.
'''
criterion = nn.BCELoss()
D_opt = torch.optim.Adam(D.parameters(), lr = 0.0002)
G_opt = torch.optim.Adam(G.parameters(), lr = 0.0002)

# Defining parameters and hyperparameters
max_epoch = 11
step = 0
n_critic = 1 # Number of time generator is trained for every iteration of discriminator training in every epoch
n_noise = 100 # Latent dimension of random noise fed as input to generator

D_labels = to_cuda(torch.ones(batch_size)) 
D_fakes = to_cuda(torch.zeros(batch_size)) 


# Training Loop
'''
idx, (images, labels) store the index, input images and output labels of every image in the batch

'''
for epoch in range(max_epoch):
    for idx, (images, labels) in enumerate(data_loader):
        step += 1
        # Training Discriminator
        x = to_cuda(images)
        x_outputs = D(x)
        D_x_loss = criterion(x_outputs, D_labels) #Training discriminator to output 1 for real image

        z = to_cuda(torch.randn(batch_size, n_noise))
        z_outputs = D(G(z))
        D_z_loss = criterion(z_outputs, D_fakes) #Training discriminator to output 0 for generator output image
        D_loss = D_x_loss + D_z_loss
        
        D.zero_grad() # make the gradients zero for every batch before backpropagation
        D_loss.backward() # Evaluate the gradients with binary cross-entropy loss
        D_opt.step() # Backpropagation with Adam optimizer to update discriminator weights for every layer

        if step % n_critic == 0:
            # Training Generator
            z = to_cuda(torch.randn(batch_size, n_noise)) 
            z_outputs = D(G(z))
            G_loss = criterion(z_outputs, D_labels)

            D.zero_grad() 
            G.zero_grad()
            G_loss.backward() 
            G_opt.step() # Backpropagation with Adam optimizer to update generator weights for every layer
        
        # Print Losses afer every 1000 batch processing
        if step % 1000 == 0:
            print('Epoch: {}/{}, Step: {}, D Loss: {}, G Loss: {}'.format(epoch, max_epoch, step, D_loss.data[0], G_loss.data[0]))
        
        # Saving output images after every 5 epochs          
        if epoch % 5 == 0:
            G.eval()
            img = get_output_image()
            scipy.misc.imsave('sample/epoch_{}_type1.jpg'.format(epoch), img)
            G.train()





