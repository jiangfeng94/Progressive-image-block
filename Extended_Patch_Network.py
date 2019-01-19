import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import torchvision
from torch.autograd import Variable
import numpy as np
from utils import weights_init
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [  nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
                    nn.InstanceNorm2d(out_size),
                    nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        def block(in_,out_,normalization=True):
            layers = [nn.Conv2d(in_, out_, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.dis =nn.Sequential(
            *block(3*2,64,normalization=False),
            *block(64,128),
            *block(128,256),
            #*block(256,512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )
    def forward(self,img_A,img_B):
        img_input =torch.cat((img_A,img_B),1)
        out =self.dis(img_input)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down1 = UNetDown(3, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, normalize=False,dropout=0.5)
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256, dropout=0.5)
        self.up4 = UNetUp(512, 128, dropout=0.5)
        self.up5 = UNetUp(256, 64)
        self.finallayer =nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1,0,1,0)),
            nn.Conv2d(128,3,4,padding=1),
            nn.Tanh()
        )
    def forward(self,img):
        d1 = self.down1(img)    
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        out = self.finallayer(u5)
        return out

class EPN():
    def __init__(self,data,in_size,out_size,config,epochs):
        self.data =data
        self.input_size =in_size
        self.output_size =out_size
        self.config=config
        self.epochs=epochs
    def train(self):
        gan_loss =torch.nn.MSELoss()
        pixel_loss =torch.nn.L1Loss()
        lambda_pixel = 100
        Net_G =Generator()
        Net_D =Discriminator()
        Net_G.cuda()
        Net_D.cuda()
        gan_loss.cuda()
        pixel_loss.cuda()
        #Net_G.apply(weights_init)
        #Net_D.apply(weights_init)
        optimizer_G = torch.optim.Adam(Net_G.parameters(), lr=self.config['lr'], betas=(self.config['b1'], self.config['b2']))
        optimizer_D = torch.optim.Adam(Net_D.parameters(), lr=self.config['lr'], betas=(self.config['b1'], self.config['b2']))
        FloatTensor=torch.cuda.FloatTensor
        patch = (1, self.input_size//4, self.input_size//4)
        for epoch in range(self.epochs):
            for i,(imgs,_) in enumerate(self.data):
                imgs_A_ =imgs[:,:,128-self.input_size//2:128+self.input_size//2,128-self.input_size//2:128+self.input_size//2]
                imgs_B =imgs[:,:,128-self.output_size//2:128+self.output_size//2,128-self.output_size//2:128+self.output_size//2]
                imgs_A = torch.zeros(imgs_B.size())
                imgs_A[:,:,self.output_size//2-self.input_size//2:self.output_size//2+self.input_size//2,self.output_size//2-self.input_size//2:self.output_size//2+self.input_size//2]=imgs_A_
                real_A = Variable(imgs_A.type(FloatTensor))
                real_B = Variable(imgs_B.type(FloatTensor))   
                valid = Variable(FloatTensor(np.ones((self.config['batchsize'], *patch))), requires_grad=False)
                fake = Variable(FloatTensor(np.zeros((self.config['batchsize'], *patch))), requires_grad=False)
                optimizer_G.zero_grad()
                # GAN loss
                fake_B = Net_G(real_A)
                loss_gan = gan_loss(Net_D(fake_B, real_A), valid)
                # Pixel-wise loss
                center_img_B =fake_B[:,:,self.output_size//2-self.input_size//2:self.output_size//2+self.input_size//2,self.output_size//2-self.input_size//2:self.output_size//2+self.input_size//2]
                center_img_A =imgs_A[:,:,self.output_size//2-self.input_size//2:self.output_size//2+self.input_size//2,self.output_size//2-self.input_size//2:self.output_size//2+self.input_size//2]
                
                loss_pixel = pixel_loss(fake_B, real_B) +pixel_loss(center_img_B.cuda(),center_img_A.cuda())
                g_loss =loss_gan +lambda_pixel * loss_pixel
                g_loss.backward()
                optimizer_G.step()
                optimizer_D.zero_grad()
                d_loss_real =gan_loss(Net_D(real_B,real_A),valid)
                d_loss_fake =gan_loss(Net_D(fake_B.detach(), real_A),fake)
                d_loss =0.5 *(d_loss_real+d_loss_fake)
                d_loss.backward()
                optimizer_D.step()
                if i % 100 == 0:
                    if not os.path.exists('output/EPN_%d'%(self.output_size)): 
                        os.mkdir('output/EPN_%d'%(self.output_size))
                    print ("[Epoch %d] [Batch %d/%d] [D loss: %f] [G loss: %f] " % (epoch, i, len(self.data),d_loss.item(), g_loss.item()))
                    torchvision.utils.save_image(real_A.data, 'output/EPN_%d/%d_A.png' % (self.output_size,epoch * len(self.data) + i), nrow=4, normalize=True)
                    torchvision.utils.save_image(real_B.data, 'output/EPN_%d/%d_B.png' % (self.output_size,epoch * len(self.data) + i), nrow=4, normalize=True)
                    torchvision.utils.save_image(fake_B.data, 'output/EPN_%d/%d_B_fake.png' % (self.output_size,epoch * len(self.data) + i), nrow=4, normalize=True)
            if not os.path.exists('model/EPN_%d'%(self.output_size)): 
                    os.mkdir('model/EPN_%d'%(self.output_size))
            torch.save(Net_D, 'model/EPN_%d/d_1_ep%d.pkl'%(self.output_size,epoch))
            torch.save(Net_G, 'model/EPN_%d/g_1_ep%d.pkl'%(self.output_size,epoch))