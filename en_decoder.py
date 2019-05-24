from model import sample_z
import torch
import torch.nn as nn
dtype = torch.float
device = torch.device('cuda:1')
class _netG(nn.Module):
    def __init__(self, opt,img_height,img_width):
        super(_netG, self).__init__()
        self.ngpu = opt.ngpu
        self.height = img_height
        self.width = img_width
        self.main = nn.Sequential(
            # input is (nc) x 256 x 528
            nn.Conv2d(4,opt.nef,4,2,1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef) x 128 x 264
            nn.Conv2d(opt.nef,opt.nef,4,2,1, bias=False),
            nn.BatchNorm2d(opt.nef),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef) x 64 x 132
            nn.Conv2d(opt.nef,opt.nef*2,4,2,1, bias=False),
            nn.BatchNorm2d(opt.nef*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*2) x 32 x 66
            nn.Conv2d(opt.nef*2,opt.nef*4,4,2,1, bias=False),
            nn.BatchNorm2d(opt.nef*4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*4) x 16 x 33
            nn.Conv2d(opt.nef*4,opt.nef*8,4,2,1, bias=False),
            nn.BatchNorm2d(opt.nef*8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*8) x 8 x 16
            nn.Conv2d(opt.nef*8,opt.nBottleneck,4, bias=False),
            # tate size: (nBottleneck) x 1 x 1
            nn.BatchNorm2d(opt.nBottleneck),
            nn.LeakyReLU(0.2, inplace=True),
            # input is Bottleneck, going into a convolution
            nn.ConvTranspose2d(opt.nBottleneck, opt.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 64
            nn.ConvTranspose2d(opt.ngf, opt.nc, (4,8), 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input,test):

        if test == True:
            input_noise = sample_z(1,self.height,self.width)
        else:
            input_noise = sample_z(16,self.height,self.width)

        input_noise.to(device,dtype)
        input_noise = input_noise.cuda()
        if isinstance(input.data,torch.cuda.FloatTensor) and self.ngpu>1:
            input = torch.cat([input.double(),input_noise],dim=1).float()
            output = nn.parallel.data_parallel(self.main,input,range(self.ngpu))
        else:
            output = self.main((torch.cat([input.double(),input_noise],dim=1).float()))
        return output
