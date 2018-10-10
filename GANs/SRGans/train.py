import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset


from model import *

import numpy as np 
from PIL import Image
import os
import sys

os.makedirs('images', exist_ok = True)
os.makedirs('saved_models', exist_ok = True)

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
    
def to_cuda(x):
    return x.to(device)

#Parameters and hyperparameters
n_epochs = 200
batch_size = 16
lr = 0.0002
upscale_factor = 4 
imageSize = 32 #cifar 10

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

def to_cuda(x):
    return x.to(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

generator = to_cuda(Generator(upscale_factor))
discriminator = to_cuda(Discriminator())

feature_extractor = to_cuda(FeatureExtractor())

criterion_content = nn.MSELoss()
criterion_adversarial = nn.BCELoss()

# generator.apply(weights_init_normal)
# discriminator.apply(weights_init_normal)

generator_optimiser = optim.Adam(generator.parameters(), lr = lr)
discriminator_optimiser = optim.Adam(discriminator.parameters(), lr = lr)

patch_h, patch_w = int(imageSize/16), int(imageSize/16)
patch = (batch_size, 1, patch_h, patch_w)

#Adversarial ground truths
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
valid = Variable(Tensor(np.ones(patch)), requires_grad = False)
fake = Variable(Tensor(np.zeros(patch)), requires_grad = False)

lr_dim = imageSize//upscale_factor

input_lr = Tensor(batch_size, 3, lr_dim, lr_dim)
input_hr = Tensor(batch_size, 3, imageSize, imageSize)

lr_transform = transforms.Compose(
	[transforms.Resize((lr_dim,lr_dim), Image.BICUBIC), 
	transforms.ToTensor(), 
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

hr_transform = transforms.Compose([
	transforms.Resize((imageSize,imageSize), Image.BICUBIC),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform = transforms.Compose([transforms.RandomCrop(imageSize*upscale_factor),
	transforms.ToTensor()
	])


dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform
                                        )
dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

D_labels = to_cuda(torch.ones(batch_size)) 
D_fakes = to_cuda(torch.zeros(batch_size)) 

for epochs in range(n_epochs):
	for i, img in enumerate(dataloader):

		image_lr = Variable(input_lr.copy(imgs['lr']))
		image_hr = Variable(input_hr.copy(imgs['hr']))

		#Train Generator
		gen_out = discriminator(generator(image_lr))
		gen_loss = criterion_adversarial(gen_out, valid)

		gen_features = feature_extractor(gen_hr)
		real_features = Variable(feature_extractor(image_hr).data, requires_grad=False)
		feature_loss = criterion_content(gen_features, real_features)

		total_generator_loss = feature_loss + gen_loss
		total_generator_loss.backward()
		generator_optimiser.step()

		#train discriminator
		discriminator.zero_grad()
		dis_out = discriminator(image_hr)
		loss_real = criterion_adversarial(dis_out, valid)
		loss_fake = criterion_adversarial(discriminator(gen_out.detach()), fake)

		total_discriminator_loss = 0.5*(loss_fake+loss_real)
		total_discriminator_loss.backward()
		total_discriminator_loss.step()

		print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                                                            (epoch, opt.n_epochs, i, len(dataloader),
                                                            loss_D.item(), loss_G.item()))
		batches_done = epoch * len(dataloader) + i
		if batches_done % opt.sample_interval == 0:
            # Save image sample
			save_image(torch.cat((gen_hr.data, imgs_hr.data), -2),
				'images/%d.png' % batches_done, normalize=True)



	if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
		torch.save(generator.state_dict(), 'saved_models/generator_%d.pth' % epoch)
		torch.save(discriminator.state_dict(), 'saved_models/discriminator_%d.pth' % epoch)










