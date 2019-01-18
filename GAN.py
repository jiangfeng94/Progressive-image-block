import torch.nn as nn
import torch
import os
import torchvision
from torch.autograd import Variable
import numpy as np
class Generator(nn.Module):
    def __init__(self,input_size,batchsize):
        super(Generator,self).__init__()
        self.img_size =input_size
        self.init_size =self.img_size//4
        self.z_dim =100
        self.channels =3
        self.batchsize =batchsize
        self.l1 = nn.Sequential(nn.Linear(self.z_dim,128*self.init_size**2))
        self.gen =nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
    def forward(self,z):
        out =self.l1(z)
        out =out.view(self.batchsize,128,self.init_size,self.init_size)
        return self.gen(out)

class Discriminator(nn.Module):
    def __init__(self,input_size):
        self.img_size =input_size
        self.channels =3
        super(Discriminator,self).__init__()
        self.dis =nn.Sequential(
            nn.Linear(self.img_size*self.img_size*self.channels,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,1)
        )
    def forward(self,img):
        img =img.view(img.shape[0],-1)
        return self.dis(img)

class WGAN():
    def __init__(self,D,G,data,config,init_size):
        self.D = D
        self.G = G
        self.data =data
        self.config =config
        self.gamma = 0.75
        self.lambda_k = 0.001
        self.k = 0.
        self.input_size =init_size
    def compute_gradient_penalty(self,real_img, fake_img):
        FloatTensor=torch.cuda.FloatTensor
        alpha = FloatTensor(np.random.random((self.config["batchsize"],1,1,1)))
        interpolates =(alpha*real_img +(1-alpha)*fake_img).requires_grad_(True)
        d_interpolates=self.D(interpolates)
        fake =Variable(FloatTensor(self.config["batchsize"],1).fill_(1.0),requires_grad=False)
        gradients =torch.autograd.grad(
            outputs =d_interpolates,
            inputs =interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    def train(self):
        self.D.cuda()
        self.G.cuda()
        torch.load('model/g_0.pkl')
        torch.load('model/d_0.pkl')
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=self.config['lr'], betas=(self.config['b1'], self.config['b2']))
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=self.config['lr'], betas=(self.config['b1'], self.config['b2']))
        FloatTensor=torch.cuda.FloatTensor
        for epoch in range(100):
            for i,(imgs,_) in enumerate(self.data):
                crop_imgs =imgs[:,:,self.input_size*15//2:self.input_size*17//2,self.input_size*15//2:self.input_size*17//2]
                real_imgs =Variable(crop_imgs.type(FloatTensor))
                z = Variable(FloatTensor(np.random.normal(0, 1, (self.config['batchsize'], self.config['z_dim']))))

                self.optimizer_D.zero_grad()
                fake_imgs =self.G(z)
                gradient_penalty =self.compute_gradient_penalty( real_imgs.data, fake_imgs.data)
                loss_D = -torch.mean(self.D(real_imgs)) + torch.mean(self.D(fake_imgs))+10*gradient_penalty
                loss_D.backward()
                self.optimizer_D.step()

                self.optimizer_G.zero_grad()
                gen_imgs =self.G(z)
                loss_G =-torch.mean(self.D(gen_imgs))
                loss_G.backward()
                self.optimizer_G.step()
                print ("[Epoch %d] [Batch %d/%d] [D loss: %f] [G loss: %f] " % (epoch, i, len( self.data),
                                                            loss_D.item(), loss_G.item()))
                if i % 100 == 0:
                    if not os.path.exists('output'):
                        os.mkdir('output')
                    torchvision.utils.save_image(real_imgs.data, 'output/%d_crop.png' % (epoch * len(self.data) + i), nrow=4, normalize=True)
                    torchvision.utils.save_image(gen_imgs.data, 'output/%d.png' % (epoch * len(self.data) + i), nrow=4, normalize=True)
                    torch.save(self.D, 'model/d_0.pkl')
                    torch.save(self.G, 'model/g_0.pkl')
