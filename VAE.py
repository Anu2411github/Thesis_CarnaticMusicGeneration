import torch.nn as nn
import torch
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim, nef=512, nc=1):
        """
        latent_dim: size of the latent vector
        nef: number of feature maps in the biggest conv layer of the encoder
        nc: number of channels in input
        """
        super(Encoder, self).__init__()
        self.nef = nef
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(nc, nef // 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(nef // 32, nef // 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef // 16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(nef // 16, nef // 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef // 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(nef // 8, nef // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef // 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(nef // 4, nef // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef // 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layer6 = nn.Sequential(
            nn.Conv2d(nef // 2, nef, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.fc_mu = nn.Linear(nef * 3 * 3, latent_dim)
        self.fc_logvar = nn.Linear(nef * 3 * 3, latent_dim)

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, ndf=512, nc=1):
        """
        latent_dim: size of the latent vector
        ndf: number of feature maps in the biggest conv layer of the decoder
        nc: number of channels in output
        """
        super(Decoder, self).__init__()
        self.ndf = ndf
        
        self.fc = nn.Linear(latent_dim, ndf * 3 * 3)
        
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(ndf, ndf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf // 2),
            nn.ReLU(True)
        )
        
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(ndf // 2, ndf // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf // 4),
            nn.ReLU(True)
        )
        
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(ndf // 4, ndf // 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf // 8),
            nn.ReLU(True)
        )
        
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ndf // 8, ndf // 16, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ndf // 16),
            nn.ReLU(True)
        )
        
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(ndf // 16, ndf // 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf // 32),
            nn.ReLU(True)
        )
        
        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(ndf // 32, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        x = self.fc(input)
        x = x.view(x.size(0), self.ndf, 3, 3)  
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        output = self.layer6(x)
        return output

class VAE(nn.Module):
    def __init__(self, latent_dim, nef=512, ndf=512, nc=1):
        """
        VAE model that includes an encoder and a decoder.

        latent_dim: size of the latent vector
        nef: number of feature maps in the biggest conv layer of the encoder
        ndf: number of feature maps in the biggest conv layer of the decoder
        nc: number of channels in input/output
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, nef, nc)
        self.decoder = Decoder(latent_dim, ndf, nc)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        
        mu: Mean of the latent Gaussian
        logvar: Log variance of the latent Gaussian
        """
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)  # Random normal variable
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass through the VAE.

        x: Input tensor of shape (batch_size, nc, height, width)
        """
        mu, logvar = self.encoder(x)
        
        z = self.reparameterize(mu, logvar)
        
        recon_x = self.decoder(z)
        
        return recon_x, mu, logvar
    
if __name__ == '__main__':
    latent_dim = 128
    nef = 512
    ndf = 512
    nc = 1

    vae = VAE(latent_dim=latent_dim, nef=nef, ndf=ndf, nc=nc)
    vae = vae.to('cuda')
    input_tensor = torch.randn(16, nc, 200, 200)  # Batch size of 16, nc channels, 200x200 image
    input_tensor = input_tensor.to('cuda')
    reconstructed, mu, logvar = vae(input_tensor)
    print(reconstructed.shape)
