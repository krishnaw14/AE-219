import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as functional
from torchvision.models import vgg19

class FeatureExtractor(nn.Module):

	def __init__(self, feature_layer = 12):
		super(FeatureExtractor, self).__init__()

		vgg19_model = vgg19(pretrained = True)
		self.feature_extractor = nn.Sequential(
			*list(vgg19_model.features.children())[:feature_layer]
			)

	def forward(self, image):
		image_features = self.feature_extractor(image)
		return image_features

class ResidualBlock(nn.Module):

	def __init__(self, in_features = 64, n=64, s=1, f=3, p=1):
		super(ResidualBlock, self).__init__()

		self.layer1 = nn.Sequential(
			nn.Conv2d(in_features, n, f, stride = s, padding = p),
			nn.BatchNorm2d(n),
			nn.ReLU(),
			)
		self.layer2 = nn.Sequential(
			nn.Conv2d(n, n, f, stride = s, padding = p),
			nn.BatchNorm2d(n),
			)

	def forward(self, x):
		conv_output = self.layer1(x)
		conv_output = self.layer2(conv_output)

		output = conv_output + x #skip connection
		return output

class UpsampleBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(UpsampleBlock, self).__init__()
		self.upsampling_layer = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, 3, stride = 1, padding = 1),
			nn.BatchNorm2d(out_channels),
			nn.PixelShuffle(2),
			nn.ReLU(inplace=True),
			)
	def forward(self, x):
		upsampled_output = self.upsampling_layer(x)
		return upsampled_output

class Generator(nn.Module):

	def __init__(self, upsample_factor, n_residual_blocks = 16, in_channels = 3, out_channels = 3):
		super(Generator, self).__init__()

		self.n_residual_blocks = n_residual_blocks
		self.upsample_factor = upsample_factor

		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels, 64, 9, stride = 1, padding = 4),
			nn.ReLU(),
			)
		residual_blocks = []
		for i in range(n_residual_blocks):
			residual_blocks.append(ResidualBlock(64))
		self.layer2 = nn.Sequential(*residual_blocks)

		self.layer3 = nn.Sequential(
			nn.Conv2d(64, 64, 3, stride = 1, padding = 1),
			nn.BatchNorm2d(64),
			)

		upsample_blocks = []
		for i in range(self.upsample_factor//2):
			upsample_blocks.append(UpsampleBlock(64,256))
		self.layer4 = nn.Sequential(*upsample_blocks)

		self.layer5 = nn.Sequential(
			nn.Conv2d(64, out_channels, 9, stride = 1, padding = 4),
			nn.Tanh(),
			)

	def forward(self, x):
		layer1_output = self.layer1(x)
		output = self.layer2(layer1_output)
		output = self.layer3(output)
		output = torch.add(layer1_output, output) #Skip Connection
		output = self.layer4(output)
		output = self.layer5(output)

		return output

class Discriminator(nn.Module):

	def __init__(self):
		super(Discriminator, self).__init__()

		self.layer1 = nn.Sequential(
			nn.Conv2d(3, 64, 3, stride = 1, padding = 1),
			nn.LeakyReLU(0.2, inplace = True),
			)
		self.layer2 = nn.Sequential(
			nn.Conv2d(64, 64, 3, stride = 2, padding = 1),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2, inplace = True),
			)
		self.layer3 = nn.Sequential(
			nn.Conv2d(64, 128, 3, stride = 1, padding = 1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace = True),
			)
		self.layer4 = nn.Sequential(
			nn.Conv2d(128, 128, 3, stride = 2, padding = 1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace = True),
			)
		self.layer5 = nn.Sequential(
			nn.Conv2d(128, 256, 3, stride = 1, padding = 1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, inplace = True),
			)
		self.layer6 = nn.Sequential(
			nn.Conv2d(256, 256, 3, stride = 2, padding = 1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, inplace = True),
			)
		self.layer7 = nn.Sequential(
			nn.Conv2d(256, 512, 3, stride =1, padding = 1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, inplace = True),
			)
		self.layer8 = nn.Sequential(
			nn.Conv2d(512, 512, 3, stride = 2, padding = 1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, inplace = True),
			)

		self.layer9 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(512, 1024, 1),
			nn.LeakyReLU(0.2, inplace = True), 
			nn.Conv2d(1024, 1, 1)
			)

	def forward(self, x):

		output = self.layer1(x)
		output = self.layer2(output)
		output = self.layer3(output)
		output = self.layer4(output)
		output = self.layer5(output)
		output = self.layer6(output)
		output = self.layer7(output)
		output = self.layer8(output)
		output = self.layer9(output)

		return output
