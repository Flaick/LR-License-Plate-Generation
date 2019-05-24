import torch
import math
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
dtype = torch.float
device = torch.device('cuda:3')
def sample_z(bs ,m, n):
	return torch.from_numpy(np.random.uniform(-1., 1., size=[bs,1, m, n])).double()#generate a noise 1*1*m*n
class Generator(nn.Module):
    def __init__(self,img_height,img_width):
        super(Generator,self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=9, padding=4),#)int(((img_height+1)/1-img_height+9)/2)),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64,img_height,img_width)
        self.block3 = ResidualBlock(64,img_height,img_width)
        self.block4 = ResidualBlock(64,img_height,img_width)
        self.block5 = ResidualBlock(64,img_height,img_width)
        self.block6 = ResidualBlock(64,img_height,img_width)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4, stride = 2,padding=1),
            nn.PReLU()
        )#change the size of the image into 1/2
        self.block8 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride = 1,padding=1),
            nn.PReLU()
        )
        self.block9 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride = 1,padding=1),
        )
        self.height=img_height
        self.width=img_width
    def forward(self, x, test):
        if test == True:
            input_noise = sample_z(1,self.height,self.width)
        else:
            input_noise = sample_z(16,self.height,self.width)
        input_noise.to(device,dtype)
        input_noise = input_noise.cuda()
        block1 = self.block1(torch.cat([x.double(),input_noise],dim=1).float())
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)

        block8 = self.block8(block7)
        block9 = self.block9(block8)
        #block8 = self.block8(block1 + block7)

        return (torch.tanh(block9) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self,opt):
        super(Discriminator,self).__init__()
        self.ngpu = opt.ngpu
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),


            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if isinstance(x.data,torch.cuda.FloatTensor) and self.ngpu>1:
            output = nn.parallel.data_parallel(self.net,x,range(self.ngpu))
        else:
            output = self.net(x)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, channels,img_height,img_width):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)#int(((img_height+1)/1-img_height+3)/2))
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)#int(((img_height+1)/1-img_height+3)/2))
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual
def weights_init(m):
    if type(m)==nn.Conv2d:
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)
def weights_init_(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
