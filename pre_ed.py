import cv2
import argparse
import torchvision.transforms as transforms
import os
from model import weights_init_
from math import log10
from en_decoder import _netG
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import MyDataset, ToTensor
import argparse
parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--num_epochs', default=10000, type=int, help='train epoch number')
parser.add_argument('--nThreads',default=4, type=int, help='train epoch number')
parser.add_argument('--ngpu',default=1,type=int,help="number of gpu used")
parser.add_argument('--nef',type=int,default=64,help='of encoder filters in first conv layer')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--nBottleneck',type=int,default=4000)
parser.add_argument('--ngf', type=int, default=64)
opt = parser.parse_args()
NUM_EPOCHS = opt.num_epochs
torch.cuda.set_device(3)
train_data = MyDataset('./train',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))#has 2 folders which respectively have HR and LR images for training
train_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=16,
                                          shuffle=True,
                                          num_workers=opt.nThreads, drop_last=True)


dtype = torch.float
device = torch.device('cuda:3')


netG = _netG(opt,128,264)
netG.apply(weights_init_)
criterion = torch.nn.MSELoss()
if torch.cuda.is_available():
    netG.cuda()
    criterion.cuda()
optimizerG = optim.Adam(netG.parameters(),lr=0.0002,betas=(0.5,0.999))
for epoch in range(1,NUM_EPOCHS+1):
    for LR, HR, half_HR in train_loader:
        LR,HR,halt_HR = LR.to(device=device,dtype=dtype),HR.to(device=device,dtype=dtype),half_HR.to(device=device,dtype=dtype)
        if torch.cuda.is_available():
            HR = HR.cuda()
            half_HR = half_HR.cuda()
        netG.zero_grad()
        out = netG(HR,test=False)
        loss = criterion(out,half_HR)
        loss.backward()
        optimizerG.step()
        print(loss.data)
    print(epoch)
    if epoch%5==0:
        torch.save(netG.state_dict(),"./encoder_decoder_models/epoch_%d.pth"%(epoch))
        img = cv2.imread("./test.png")
        img = img.transpose((2,0,1))
        img_tensor = torch.from_numpy(img).float()
        transforms_test = transforms.Compose([transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        img_tensor = transforms_test(img_tensor)
        img_tensor = img_tensor.expand([1,3,128,264]).to(device,dtype)
        out = netG(img_tensor,test=True)
        out = torch.squeeze(out)
        out = out.cpu().detach()#.numpy().transpose((1,2,0))
        output_name = "./ed_test_result/"+str(epoch)+".png"
        #cv2.imwrite(output_name,out)
        utils.save_image(out,output_name,normalize=True)
