import torch
import torchvision.utils as utils
from en_decoder import _netG
import argparse
from model import Discriminator ,weights_init_, weights_init
from data import MyDataset
import torchvision.transforms as transforms
parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--num_epochs', default=10000, type=int, help='train epoch number')
parser.add_argument('--nThreads',default=4, type=int, help='train epoch number')
parser.add_argument('--ngpu',default=1,type=int,help="number of gpu used")
parser.add_argument('--nef',type=int,default=64,help='of encoder filters in first conv layer')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--nBottleneck',type=int,default=4000)
parser.add_argument('--ngf', type=int, default=64)
opt = parser.parse_args()
torch.cuda.set_device(0)

dtype = torch.float
device = torch.device('cuda:0')
device_model = torch.device('cuda:0')

netG = _netG(opt,128,264).to(device_model,dtype)#height:128,width:264
netG.apply(weights_init_)
netG.load_state_dict(torch.load("./save_models_G/netG_epoch_844.pth"))


val_data = MyDataset('./test_ed',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))#has 2 folders which respectively have HR and LR images for val
val_loader = torch.utils.data.DataLoader(val_data,
                                          batch_size=16,
                                          shuffle=True,
                                          num_workers=opt.nThreads, drop_last=True)

for ind ,(LR, HR, half_HR) in enumerate(val_loader):


    LR,HR,half_HR = LR.to(device=device,dtype=dtype),HR.to(device=device,dtype=dtype),half_HR.to(device=device,dtype=dtype)
    if torch.cuda.is_available():
        HR = HR.cuda()
        half_HR = half_HR.cuda()
        LR = LR.cuda()


    fake_lR = netG(HR,test=True)
    file_LR = "./produce_res/LR/"+str(ind)+".png"
    file_HR = "./produce_res/HR/"+str(ind)+".png"
    file_HR_12 = "./produce_res/Half/"+str(ind)+".png"
    utils.save_image(utils.make_grid(fake_lR),file_LR)
    utils.save_image(utils.make_grid(half_HR),file_HR_12)
    utils.save_image(utils.make_grid(HR),file_HR)
