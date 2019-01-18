import argparse
import torch
import torchvision
from GAN import Generator,Discriminator,WGAN
from dataset import get_dataloader
import numpy as np
parse =argparse.ArgumentParser()
parse.add_argument('--batchsize',type=int,default=16)
parse.add_argument('--lr',type=float,default=0.0002)
parse.add_argument('--b1',type=float,default=0.5)
parse.add_argument('--b2',type=float,default=0.999)
parse.add_argument('--z_dim',type=int,default=100)
parse.add_argument('--channels',type=int,default=1)
parse.add_argument('--img_size',type=int,default=256)
parse.add_argument('--img_path',type=str,default='D:\img_align_celeba')
opt =parse.parse_args()



def test():
    g_0 = torch.load('model/g_0.pkl')
    z = torch.autograd.Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (opt.batchsize, opt.z_dim))))
    output = g_0(z)
    torchvision.utils.save_image(output.data, 'output/test.png' , nrow=4, normalize=True)

if __name__ == "__main__":
    data =get_dataloader(opt.img_size,opt.batchsize,opt.img_path)
    config = {k:v for k,v in opt._get_kwargs()}
    init_size =opt.img_size//8
    g_0 = Generator(init_size,opt.batchsize)
    d_0 = Discriminator(init_size)
    gan = WGAN(d_0, g_0, data, config,init_size)
    gan.train()
    test()


    
            
