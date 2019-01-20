import argparse
import torch
import torchvision
import numpy as np
from dataset import get_dataloader
from GAN import WGAN
from Extended_Patch_Network import EPN
parse =argparse.ArgumentParser()
parse.add_argument('--batchsize',type=int,default=16)
parse.add_argument('--lr',type=float,default=0.00015)
parse.add_argument('--b1',type=float,default=0.5)
parse.add_argument('--b2',type=float,default=0.999)
parse.add_argument('--z_dim',type=int,default=100)
parse.add_argument('--channels',type=int,default=1)
parse.add_argument('--img_size',type=int,default=256)
parse.add_argument('--img_path',type=str,default='D:\img_align_celeba')
opt =parse.parse_args()



def test():
    #GAN generate center patch 32*32*3
    g_0 = torch.load('model/gan_0/g_0_ep16.pkl')
    z = torch.autograd.Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (opt.batchsize, opt.z_dim))))
    output_0 = g_0(z)
    torchvision.utils.save_image(output_0.data, 'output/test_32.png' , nrow=4, normalize=True)
    #CEN generate 64*64*3
    g_1 = torch.load('model/EPN_64/g_1_ep0.pkl')
    intput_1 =torch.autograd.Variable(torch.cuda.FloatTensor(np.zeros([16,3,64,64])))
    intput_1[:,:,16:48,16:48]=output_0
    output_1 = g_1(intput_1)
    torchvision.utils.save_image(output_1.data, 'output/test_64.png' , nrow=4, normalize=True)
    #CEN generate 64*64*3
    g_2 = torch.load('model/EPN_128/g_1_ep0.pkl')
    intput_2 =torch.autograd.Variable(torch.cuda.FloatTensor(np.zeros([16,3,128,128])))
    intput_2[:,:,32:96,32:96]=output_1
    output_2 = g_2(intput_2)
    torchvision.utils.save_image(output_2.data, 'output/test_128.png' , nrow=4, normalize=True)

if __name__ == "__main__":
    data =get_dataloader(opt.img_size,opt.batchsize,opt.img_path)
    config = {k:v for k,v in opt._get_kwargs()}
    progressive_size =[32,64,128]
    
    #gan = WGAN(data,config,progressive_size[0])
    #gan.train()
    #epn_1 =EPN(data,progressive_size[0],progressive_size[1],config,1)
    #epn_1.train()
    #epn_2 =EPN(data,progressive_size[1],progressive_size[2],config,1)
    #epn_2.train()
    test()



    
            
