#import argparse
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
   
import datasetloader
from models import CycleGenerator
class Opt(object):
    def __init__(self):
        return
opt = Opt()


opt.generator_A2B = 'output24/netG_A2B.pth'
opt.generator_B2A = 'output24/netG_B2A.pth'
opt.device = 'cuda:0' #opt0: device
opt.dataset = 'h2z'
opt.train_dir = 'test'
opt.data_dir = './datasets/'
opt.batch_size = 1
opt.size = 256
opt.unaligned = 0
opt.noise = 0
opt.A_nc =3
opt.B_nc = 3
opt.cuda = 1
opt.lr = 0.0002
opt.epoch_count = 0
opt.n_epochs = 100 
opt.n_epochs_decay = 100 
opt.port = 35850
opt.test = 1
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = CycleGenerator(opt.A_nc, opt.B_nc)
netG_B2A = CycleGenerator(opt.B_nc, opt.A_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts
netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batch_size, opt.A_nc, opt.size, opt.size)
input_B = Tensor(opt.batch_size, opt.B_nc, opt.size, opt.size)

# Dataset loader
data_loader = datasetloader.DatasetLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(dataset)
###################################

###### Testing######

# Create output dirs if they don't exist
if not os.path.exists('output24/A'):
    os.makedirs('output24/A')
if not os.path.exists('output24/B'):
    os.makedirs('output24/B')

for i, batch in enumerate(dataset):
    # Set model input
    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))


    # Generate output
    fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
    fake_A = 0.5*(netG_B2A(real_B).data + 1.0)
    real_A = 0.5*(real_A.data + 1.0)
    real_B = 0.5*(real_B.data + 1.0)

    # Save image files
    save_image(fake_A, 'output24/A/%04d.png' % (i+1))
    save_image(fake_B, 'output24/B/%04d.png' % (i+1))
    save_image(real_B, 'output24/A/%04d_real.png' % (i+1))
    save_image(real_A, 'output24/B/%04d_real.png' % (i+1))

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, dataset_size))

sys.stdout.write('\n')
