import matplotlib as mpl
mpl.use('Agg')
from utils import draw_loss
import cv2
import argparse
import torchvision.transforms as transforms
import os
from math import log10
from model import Discriminator ,weights_init_, weights_init
from en_decoder import _netG
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import MyDataset, ToTensor
#import pytorch_ssim
#from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
#from loss import GeneratorLoss
#from model import Generator, Discriminator
import argparse
parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--num_epochs', default=10000, type=int, help='train epoch number')
parser.add_argument('--nThreads',default=4, type=int, help='train epoch number')
parser.add_argument('--ngpu',default=1,type=int,help="number of gpu used")
parser.add_argument('--nef',type=int,default=64,help='of encoder filters in first conv layer')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--nBottleneck',type=int,default=4000)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ngf', type=int, default=64)
opt = parser.parse_args()
NUM_EPOCHS = opt.num_epochs
torch.cuda.set_device(0)


real_LR_label = 0
fake_LR_label = 1

dtype = torch.float
device = torch.device('cuda:0')
device_model = torch.device('cuda:3')


train_data = MyDataset('./train',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))#has 2 folders which respectively have HR and LR images for training
train_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=16,
                                          shuffle=True,
                                          num_workers=opt.nThreads, drop_last=True)

val_data = MyDataset('./val',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))#has 2 folders which respectively have HR and LR images for val
val_loader = torch.utils.data.DataLoader(val_data,
                                          batch_size=16,
                                          shuffle=True,
                                          num_workers=opt.nThreads, drop_last=True)
'''for ind,test in enumerate(train_loader):
    img_height = test.size(2)
    img_width = test.size(3)
    if ind == 1:
        break'''
netG = _netG(opt,128,264).to(device_model,dtype)#height:128,width:264
netG.apply(weights_init_)
#netG.load_state_dict(torch.load("./encoder_decoder_models/epoch_40.pth"))
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Discriminator(opt).to(device_model,dtype)
netD.apply(weights_init_)
#netD.load_state_dict(torch.load("./save_models_D/netD_epoch_100.pth"))
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
print("generator structure:",netG)
print("discriminator structure",netD)
generator_criterion = torch.nn.MSELoss()
dis_criterion = torch.nn.BCELoss()

if torch.cuda.is_available():
    netG.cuda()
    netD.cuda()
    generator_criterion.cuda()
    dis_criterion.cuda()

start = 0
end = 45
 

optimizerG = optim.Adam(netG.parameters(),lr=0.0002,betas=(0.5,0.999))
optimizerD = optim.Adam(netD.parameters(),lr=0.0002,betas=(0.5,0.999))

results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'mse': [], 'label': []}

for epoch in range(1, NUM_EPOCHS + 1):
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    Gnum = 10
    netG.train()
    netD.train()
    D_ = []
    G_ = []
    for ind, (LR,HR,half_HR) in enumerate(train_loader):
        LR, HR, half_HR = LR.to(device=device,dtype=dtype), HR.to(device=device,dtype=dtype), half_HR.to(device,dtype)
        batch_size = LR.size(0)

        running_results['batch_sizes'] += batch_size
        if torch.cuda.is_available():
            LR = LR.cuda()
            HR = HR.cuda()
            half_HR = half_HR.cuda()
            real_ = torch.full((batch_size,),real_LR_label,device=device)
            fake_ = torch.full((batch_size,),fake_LR_label,device=device)
        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################

        netD.zero_grad()
        real_label = netD(LR).view(-1)
        real_loss = dis_criterion(real_label, real_)
        real_loss.backward()

        fake_LR = netG(HR,test=False)
        fake_label = netD(fake_LR).view(-1)
        fake_loss = dis_criterion(fake_label,fake_)
        fake_loss.backward()
        d_loss = real_loss + fake_loss
        #d_loss.backward()
        optimizerD.step()
        running_results['d_loss'] += d_loss.data * batch_size
        running_results['d_score'] += real_label.mean() * batch_size

        print('[%d/%d] Loss_D: %.4f D(x): %.4f  real_loss: %.4f  fake_loss: %.4f' % (
            epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
            running_results['d_score'] / running_results['batch_sizes'],real_loss.data,fake_loss.data))
        D_.append(d_loss.data)

        ############################
        # (2) Update G network: minimize (1-D(G(z)) + Image Loss)
        ###########################
        if ind%1==0 :
            netG.zero_grad()
            fake_LR = netG(HR,test=False)
            #fake_LR = transforms_test(fake_LR)
            fake_label = netD(fake_LR).view(-1)
 
            mse_loss = generator_criterion(fake_LR, half_HR)
            dis = dis_criterion(fake_label,real_)
            g_loss = 0.3*mse_loss + 0.7*dis
            g_loss.backward()
            optimizerG.step()
            ####################################
            ##training logs
            ####################################
            running_results['g_loss'] += g_loss.data * batch_size
            running_results['g_score'] += fake_label.mean() * batch_size

            print('[%d/%d] Loss_G: %.4f D(G(z)): %.4f MSE: %.4f Dis: %.4f' % (
            epoch, NUM_EPOCHS,running_results['g_loss'] / running_results['batch_sizes'],
            running_results['g_score'] / running_results['batch_sizes'],mse_loss.data,dis.data))
            G_.append(g_loss.data)

    if epoch %1 ==0:
        Gout_path = './result_log/'
       # save model parameters
        torch.save(netG.state_dict(), './save_models_G/netG_epoch_%d.pth' % (epoch))
        torch.save(netD.state_dict(), './save_models_D/netD_epoch_%d.pth' % (epoch))
        img = cv2.imread("./test.png")
        img = img.transpose((2,0,1))
        img_tensor = torch.from_numpy(img).float()
        transforms_test = transforms.Compose([transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        img_tensor = transforms_test(img_tensor)
        img_tensor = img_tensor.expand([1,3,128,264]).to(device,dtype)
        out = netG(img_tensor,test=True)
        out = torch.squeeze(out)
        out = out.cpu().detach()#.numpy().transpose((1,2,0))
        output_name = "./test_result/"+str(epoch)+".png"
        #cv2.imwrite(output_name,out)
        utils.save_image(out,output_name,normalize=True)
        #draw loss figure
        #print("save loss figure")
        lst_iter = range(start,end)#have 45 batches in a epoch 
        print(len(D_))
        print(len(G_))
        draw_loss(lst_iter,D_,G_,epoch)
        start = start + 45
        end = end + 45
