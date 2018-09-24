import numpy as np 
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

if torch.cuda.is_available():
	cuda = True
else:
	cuda = False

class Mnist:
    def __init__(self, batch_size):
        dataset_transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])

        train_dataset = datasets.MNIST('../data', train=True, download=False, transform=dataset_transform)
        test_dataset = datasets.MNIST('../data', train=False, download=False, transform=dataset_transform)
        
        self.train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)  


def squash(u):
	squared_norm = (u**2).sum(-1, keepdim = True)
	scale = squared_norm/((1.0 + squared_norm)*torch.sqrt(squared_norm))
	v = scale*u
	return v

class ConvLayer(nn.Module):

	def __init__(self, in_channels = 1, out_channels = 256, kernel_size = 9):
		super(ConvLayer, self).__init__()

		self.conv_layer = nn.Sequential(
			nn.Conv2d(in_channels, out_channels,kernel_size, stride = 1), 
			nn.ReLU()
			)

	def forward(self, x):
		output = self.conv_layer(x)
		return output


class PrimaryCaps(nn.Module):

	def __init__(self, num_capsules = 8, in_channels = 256, out_channels = 32, kernel_size = 9):
		super(PrimaryCaps, self).__init__()

		capsule_blocks = []
		for i in range(num_capsules):
			capsule_blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride = 2, padding = 0))

		self.PrimaryCaps_layer = nn.Sequential(*capsule_blocks)

	def forward(self, x):
		u = []
		for layer in self.PrimaryCaps_layer:
			u.append(layer(x))
		u = torch.stack(u, dim = 1)
		u = u.view(x.size(0), 32*6*6, -1)
		return squash(u)


class DigitCaps(nn.Module):

	def __init__(self, num_capsules = 10, num_routes = 32*6*6, in_channels = 8, out_channels = 16):
		super(DigitCaps, self).__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.num_capsules = num_capsules
		self.num_routes = num_routes
	
		self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

	def forward(self, x):
		batch_size = x.size(0)
		x = torch.stack([x]*self.num_capsules, dim = 2).unsqueeze(4)

		W = torch.cat([self.W]*batch_size, dim = 0)
		u_hat = torch.matmul(W,x)

		bij = Variable(torch.zeros(1,self.num_routes, self.num_capsules, 1))
		if cuda:
			bij = bij.cuda()

		routing_iterations = 3
		for i in range(routing_iterations):
			cij = F.softmax(bij)
			cij = torch.cat([cij]*batch_size, dim = 0).unsqueeze(4)

			s_j = (cij*u_hat).sum(dim = 1, keepdim = True)
			v_j = squash(s_j)

			if i < routing_iterations-1:
				aij = torch.matmul(u_hat.transpose(3,4), torch.cat([v_j] * self.num_routes, dim=1))
				bij = bij + aij.squeeze(4).mean(dim=0, keepdim=True)

		return v_j.squeeze(1)



class Decoder(nn.Module):

	def __init__(self):
		super(Decoder, self).__init__()

		self.reconstruction_layers = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

	def forward(self, x, data):
		classes = torch.sqrt((x**2).sum(2))
		classes = F.softmax(classes)

		_, max_length_indices = classes.max(dim = 1)
		masked = Variable(torch.eye(10))

		if cuda:
			masked = masked.cuda()
		masked = masked.index_select(dim=0, index=max_length_indices.squeeze(1).data)
        
		reconstructions = self.reconstruction_layers((x * masked[:, :, None, None]).view(x.size(0), -1))
		reconstructions = reconstructions.view(-1, 1, 28, 28)
        
		return reconstructions, masked


class CapsuleNetwork(nn.Module):

	def __init__(self):
		super(CapsuleNetwork, self).__init__()
		self.conv_layer = ConvLayer()
		self.primary_capsule = PrimaryCaps()
		self.digit_capsule = DigitCaps()
		self.decoder = Decoder()

		self.mse_loss = nn.MSELoss()

	def forward(self, x):
		output = self.digit_capsule(
			self.primary_capsule(self.conv_layer(x))
			)
		reconstructions, masked = self.decoder(output, data)
		return output, reconstructions, masked

	def loss(self, data, output, target, reconstructions):
		margin_loss = self.margin_loss(output,target) 
		reconstruction_loss = self.reconstruction_loss(data, reconstructions)
		total_loss = margin_loss + 0.0005*reconstruction_loss
		return total_loss

	def margin_loss(self, output, target):
		batch_size = output.size(0)

		v_k = torch.sqrt((output**2).sum(dim = 2, keepdim = True))

		present_error = F.relu(0.9-v_k).view(batch_size, -1)
		absent_error = F.relu(v_k-0.1).view(batch_size, -1)

		loss = target*present_error + 0.5*(1-target)*absent_error
		loss = loss.sum(dim = 1).mean()

		return loss

	def reconstruction_loss(self, data, reconstructions):
		loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
		return loss


batch_size = 100
mnist = Mnist(batch_size)

n_epochs = 30

capsule_net = CapsuleNetwork()
if cuda:
    capsule_net = capsule_net.cuda()
optimizer = optim.Adam(capsule_net.parameters())


for epoch in range(n_epochs):
    capsule_net.train()
    train_loss = 0
    for batch_id, (data, target) in enumerate(mnist.train_loader):

        target = torch.eye(10).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        
        if batch_id % 100 == 0:
            print ("train accuracy:", sum(np.argmax(masked.data.cpu().numpy(), 1) == 
                                   np.argmax(target.data.cpu().numpy(), 1)) / float(batch_size) )
        
    print (train_loss / len(mnist.train_loader) )
        
    capsule_net.eval()
    test_loss = 0
    for batch_id, (data, target) in enumerate(mnist.test_loader):

        target = torch.eye(10).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        output, reconstructions, masked = capsule_net(data)
        loss = capsule_net.loss(data, output, target, reconstructions)

        test_loss += loss.data[0]
        
        if batch_id % 100 == 0:
            print ("test accuracy:", sum(np.argmax(masked.data.cpu().numpy(), 1) == 
                                   np.argmax(target.data.cpu().numpy(), 1)) / float(batch_size) )
    
    print (test_loss / len(mnist.test_loader))






