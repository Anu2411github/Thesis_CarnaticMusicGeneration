# Generator Code

import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, ngpu, nz=100, ngf=64, nc=1):
        """
        nz: size of the latent z vector
        ngf: size of feature maps in generator
        nc: number of channels in output
        """
        super(Generator, self).__init__()
        self.ngpu = ngpu
        
        # Layer 1: Input is Z, going into a convolution
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        )
        # State size: (ngf*8) x 4 x 4
        
        # Layer 2
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 3, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        # State size: (ngf*4) x 10 x 10
        
        # Layer 3
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 9, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        # State size: (ngf*2) x 25 x 25
        
        # Layer 4
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )
        # State size: (ngf) x 50 x 50
        
        # Layer 5
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True)
        )
        # State size: (ngf//2) x 100 x 100
        
        # Layer 6
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        output = self.layer6(x)
        return output
    
class Discriminator(nn.Module):
    def __init__(self, ngpu, nc=1, ndf=64):
        """
        nc: number of channels in input
        ndf: size of feature maps in discriminator
        """
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 200 x 200
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf) x 100 x 100
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*2) x 50 x 50
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*4) x 25 x 25
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 12 x 12
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*16) x 6 x 6
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*32) x 3 x 3
            nn.Conv2d(ndf * 32, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
            # output size: 1
        )

    def forward(self, input):
        v = self.main(input)
        return self.main(input)


def init_GAN(ngpu, nz=100, ngf=64, nc=1, ndf=64):
    netG = Generator(ngpu, nz, ngf, nc)
    netD = Discriminator(ngpu, nc, ndf)
    return netG, netD