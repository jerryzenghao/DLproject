from PIL import Image
import torch.utils.data as data
import os
import random
import torchvision.transforms as transforms
import torch
import numpy as np
class Dataset(data.Dataset):
    def __init__(self, opt):
        super(Dataset, self).__init__()
        self.dir_A = os.path.join(opt.data_dir+opt.dataset, opt.train_dir+'A')
        self.dir_B = os.path.join(opt.data_dir+opt.dataset, opt.train_dir+'B')
        self.A_paths = [x for x in sorted(os.listdir(self.dir_A))]
        self.B_paths = [x for x in sorted(os.listdir(self.dir_B))]
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.unaligned = opt.unaligned
        if opt.test == 1:
            self.transforms = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((.5,.5,.5), (.5,.5,.5))])
        else:
            self.transforms = transforms.Compose([transforms.Resize(int(opt.size*1.12),Image.BICUBIC),
                          transforms.RandomCrop(opt.size),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize((.5,.5,.5), (.5,.5,.5))])

        self.noise = opt.noise
    def __getitem__(self, index):
        # Load Image
        if self.unaligned == 1:
            A_path = os.path.join(self.dir_A, self.A_paths[index%self.A_size])
            B_path = os.path.join(self.dir_B, self.B_paths[random.randint(0,self.B_size-1)])
        else:
            A_path = os.path.join(self.dir_A, self.A_paths[index%self.A_size])
            B_path = os.path.join(self.dir_B, self.B_paths[index%self.B_size])
        
        A_img = self.transforms(Image.open(A_path).convert("RGB"))
        B_img = self.transforms(Image.open(B_path).convert("RGB"))
        if self.noise ==1:
            c, m,n =A_img.size()
            B = np.random.rand(m,n)
            B = (B-0.5)*2
            B_img = torch.from_numpy(B)
            B_img = B_img.expand(c,m,n)    
        return {'A':A_img, 'B':B_img, 'A_path':A_path, 'B_path':B_path }

    def __len__(self):
        
        return max(self.A_size, self.B_size)

        
