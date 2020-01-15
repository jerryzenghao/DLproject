import itertools
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.utils.data
import dataset
import datasetloader
import importlib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import models 
from torch.autograd import Variable
from utils import *
class Opt(object):
    def __init__(self):
        return
opt = Opt()

#importlib.reload(dataset)
opt.device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opt.dataset = 'h2z'
opt.train_dir = 'train'
opt.data_dir = './datasets/'
opt.batch_size = 1
opt.size = 256
opt.unaligned = 1
opt.noise = 0
opt.test = 0
dataset = dataset.Dataset(opt)
#Img = dataset.__getitem__(1)
#Img
opt.A_nc =3
opt.B_nc = 3
opt.cuda = 1
opt.lr = 0.0002
opt.epoch_count = 0
opt.n_epochs = 100
opt.n_epochs_decay = 100 
opt.port = 35854
opt.feature = 1
opt.lambdat = 10
opt.identity = 0
netG_A2B = models.CycleGenerator(opt.A_nc, opt.B_nc)
netG_B2A = models.CycleGenerator(opt.B_nc, opt.A_nc)
netD_A = models.CycleDiscriminator(opt.A_nc)
netD_B = models.CycleDiscriminator(opt.B_nc)


#if torch.cuda.device_count() > 1:
 #   print("Let's use", torch.cuda.device_count(), "GPUs!")  
  #3 netG_B2A = torch.nn.DataParallel(netG_B2A,device_ids=[0, 1, 2])
  #  netD_A = torch.nn.DataParallel(netD_A,device_ids=[0, 1, 2])
  #  netD_B = torch.nn.DataParallel(netD_B,device_ids=[0, 1, 2])

if opt.cuda:
    netG_A2B.to(opt.device)
    netG_B2A.to(opt.device)
    netD_A.to(opt.device)
    netD_B.to(opt.device)


    
netG_A2B.apply(models.weights_init)
netG_B2A.apply(models.weights_init)
netD_A.apply(models.weights_init)
netD_B.apply(models.weights_init)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt).lambda_rule)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt).lambda_rule)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt).lambda_rule)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batch_size, opt.A_nc, opt.size, opt.size)
input_B = Tensor(opt.batch_size, opt.B_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batch_size).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Loda data
data_loader = datasetloader.DatasetLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(dataset)
# Loss plot
#logger = Logger(opt.n_epochs + opt.n_epochs_decay, dataset_size,opt.port)
#for i, j in enumerate(dataset):
#    print(j)
# for epoch in range(1):
#     for i, batch in enumerate(dataset):
for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay):
    gamma = 0.0
    for i, batch in enumerate(dataset):
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B)*0.5*opt.identity
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A)*0.5*opt.identity

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        #loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0
        if opt.feature ==1:
            feature_reA = netD_A.lastlayer(recovered_A)
            feature_realA = netD_A.lastlayer(real_A).detach()
            loss_cycle_ABA = (1-gamma)*criterion_cycle(recovered_A, real_A)*opt.lambdat + gamma*criterion_cycle(feature_reA,feature_realA)*opt.lambdat
        else:
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*opt.lambdat
        recovered_B = netG_A2B(fake_A)
        if opt.feature == 1:
            feature_reB = netD_B.lastlayer(recovered_B)
            feature_realB = netD_B.lastlayer(real_B).detach()
            loss_cycle_BAB = (1-gamma)*criterion_cycle(recovered_B, real_B)*opt.lambdat + gamma*criterion_cycle(feature_reB,feature_realB)*opt.lambdat
        else:
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*opt.lambdat

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()
        
        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)
        
        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()
        ###################################
        # Progress report (http://localhost:8097)
        #logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                  #  'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)}, 
                 #  images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})
        print('epoch:',epoch,'iter:',i, 'loss_G_cycle',(loss_cycle_ABA + loss_cycle_BAB).data, 'loss_D', (loss_D_A + loss_D_B).data,'loss_G', loss_G.data)
    # Update learning rates
    if opt.feature ==1:
        opt.lambdat = max(opt.lambdat-0.0025,5)
        gamma = min(gamma+0.0025,0.5) 
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    torch.save(netG_A2B.state_dict(), 'output24/netG_A2B.pth')
    torch.save(netG_B2A.state_dict(), 'output24/netG_B2A.pth')
    torch.save(netD_A.state_dict(), 'output24/netD_A.pth')
    torch.save(netD_B.state_dict(), 'output24/netD_B.pth')
###################################
