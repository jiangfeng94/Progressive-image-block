import argparse
import torch
from GAN import Generator,Discriminator,WGAN
from utils import weights_init
from dataset import get_dataloader
parse =argparse.ArgumentParser()
parse.add_argument('--batchsize',type=int,default=16)
parse.add_argument('--lr',type=float,default=0.0002)
parse.add_argument('--b1',type=float,default=0.5)
parse.add_argument('--b2',type=float,default=0.999)
parse.add_argument('--z_dim',type=int,default=100)
parse.add_argument('--channels',type=int,default=1)
parse.add_argument('--img_size',type=int,default=64)
parse.add_argument('--img_path',type=str,default='D:\img_align_celeba')
opt =parse.parse_args()





if __name__ == "__main__":
    dataloader =get_dataloader(opt.img_size,opt.batchsize,opt.img_path)
    config = {k:v for k,v in opt._get_kwargs()}
    g_0 = Generator()
    d_0 = Discriminator()
    gan = WGAN(d_0, g_0, dataloader, config)
    gan.train()

    
            
