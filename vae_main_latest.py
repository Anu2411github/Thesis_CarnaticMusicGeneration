from dataset import MusicDataset
from GAN import init_GAN
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import os 
import gc
from plots import plot_losses
from generate_samples import generate_samples
import random
import numpy as np

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import gc
from plots import plot_losses


from tqdm import tqdm

import torch.nn.functional as F
from torch.nn.utils import spectral_norm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def perceptual_loss(reconstructed, original, mu, logvar, beta):
    mse_loss = nn.functional.mse_loss(reconstructed, original, reduction='mean')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse_loss + beta * kld_loss

class ConvEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(ConvEncoder, self).__init__()
        self.conv1 = spectral_norm(nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1))
        self.conv3 = spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1))
        self.conv4 = spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1))

        self.fc_mu = nn.Linear(512 * 5 * 31, latent_dim)
        self.fc_logvar = nn.Linear(512 * 5 * 31, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    

    # Define ResidualBlock class
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        # Update LayerNorm to match the actual dimensions after convolution
        self.bn1 = nn.LayerNorm([out_channels, 5, 31])
        self.bn2 = nn.LayerNorm([out_channels, 5, 31])

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

# Define SelfAttention class
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

# Now include the ConvDecoder class
class ConvDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(ConvDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 512 * 5 * 31)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

        self.res_block = ResidualBlock(512, 512)
        self.attn = SelfAttention(512)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 512, 5, 31)
        x = self.res_block(x)
        x = self.attn(x)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def train_GAN(generator, discriminator, num_epochs, dataloader, optimizer_D, optimizer_G, criterion, z_dim, model_path):
    all_losses_g = []
    all_losses_d = []

    # Outer tqdm for epochs
    epoch_progress = tqdm(range(num_epochs), desc='Epochs')
    for epoch in epoch_progress:
        g_losses = []
        d_losses = []
        
        # Inner tqdm for steps within each epoch
        step_progress = tqdm(enumerate(dataloader), desc='Steps', leave=False, total=len(dataloader))
        for i, (mel_spectrograms) in step_progress:
            mel_spectrograms = mel_spectrograms.to('cuda').unsqueeze(1)  # Add channel dimension

            # Train Discriminator
            optimizer_D.zero_grad()

            # Real data
            real_labels = torch.ones(mel_spectrograms.size(0)).to('cuda')
            real_outputs = discriminator(mel_spectrograms).view(-1)
            real_loss = criterion(real_outputs, real_labels)

            # Fake data
            z = torch.randn(mel_spectrograms.size(0), z_dim, 1, 1, device='cuda')
            fake_data = generator(z)
            fake_labels = torch.zeros(fake_data.size(0)).to('cuda')
            fake_outputs = discriminator(fake_data.detach()).view(-1)
            fake_loss = criterion(fake_outputs, fake_labels)
            
            # Total discriminator loss
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            # Recompute fake_outputs for generator's loss
            fake_outputs = discriminator(fake_data).view(-1)
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            optimizer_G.step()

            # Save losses for plotting
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        all_losses_d.extend(d_losses)
        all_losses_g.extend(g_losses)
        epoch_progress.set_postfix_str(f'D loss: {sum(d_losses)/len(d_losses):.4f}, G loss: {sum(g_losses)/len(g_losses):.4f}')

        # Generate and save samples and model
        if (epoch + 1) % 5 == 0:
            path = model_path + f'/epoch_{epoch + 1}'
            os.makedirs(path, exist_ok=True)
            os.makedirs(path+'/npy', exist_ok=True)
            os.makedirs(path+'/png', exist_ok=True)
            os.makedirs(path+'/wav', exist_ok=True)
            generate_samples(generator, z_dim, num_samples=50, path_npy=path+'/npy', path_png=path+'/png', path_wav=path+'/wav')

        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), os.path.join(model_path, f'generator_epoch_{epoch + 1}.pt'))
            torch.save(discriminator.state_dict(), os.path.join(model_path, f'discriminator_epoch_{epoch + 1}.pt'))

    return generator, discriminator, all_losses_d, all_losses_g


def train_VAE(vae, dataloader, num_epochs, optimizer, model_path):
    all_losses = []
    beta = 0.01  

    epoch_progress = tqdm(range(num_epochs), desc='Epochs')
    for epoch in epoch_progress:
        epoch_losses = []
        step_progress = tqdm(enumerate(dataloader), desc='Steps', leave=False, total=len(dataloader))
        
        for i, (mel_spectrograms) in step_progress:
            mel_spectrograms = mel_spectrograms.to('cuda').unsqueeze(1)
            
            optimizer.zero_grad()
            reconstructed, mu, logvar = vae(mel_spectrograms)
            loss = perceptual_loss(reconstructed, mel_spectrograms, mu, logvar, beta)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        all_losses.append(sum(epoch_losses) / len(epoch_losses))
        epoch_progress.set_postfix_str(f'Loss: {all_losses[-1]:.4f}')

        beta = min(1.0, beta + 0.05)

        if (epoch + 1) % 10 == 0:
            torch.save(vae.state_dict(), os.path.join(model_path, f'vae_epoch_{epoch + 1}.pt'))

    return vae, all_losses

def train_model(model_type, params, dataloader, model_path):
    if model_type == 'VAE':
        vae = VAE(latent_dim=params['z_dim']).to('cuda')
        optimizer = optim.Adam(vae.parameters(), lr=params['lr'])
        vae, losses = train_VAE(vae, dataloader, params['num_epochs'], optimizer, model_path)
        return vae, losses, optimizer
    
    elif model_type == 'GAN':
        # Skip GAN training for now
        print("Skipping GAN training")
        return None, None, None, None, None, None  # Return placeholder values

    else:
        raise ValueError(f"Model type {model_type} not recognized")

if __name__ == '__main__':
    set_seed(24)  
    
    paths = ['/home/msp/Downloads/Indian_Raga_Navee_Dataset', '/home/msp/Anusha/Thesis/dataset/CMR_full_dataset_1.0/audio', '/home/msp/Downloads/carnatic_varnam_(1)/carnatic_varnam_1.0/Audio']
    data = MusicDataset(paths, segment_length=200, target_sr=16000)
    num_epochs = 100

    VAE_params = [
        {'z_dim': 256, 'nef':64, 'ndf':64, 'num_epochs': num_epochs, 'lr': 1e-4, 'batch_size': 64},
        {'z_dim': 512, 'nef':128, 'ndf':128, 'num_epochs': num_epochs, 'lr': 1e-4, 'batch_size': 64},
        {'z_dim': 1024, 'nef':256, 'ndf':256, 'num_epochs': num_epochs, 'lr': 1e-4, 'batch_size': 32},
    ]

    GAN_params = [
        {'z_dim': 256, 'nef':64, 'ndf':64, 'num_epochs': num_epochs, 'lr_g': 0.0005, 'lr_d': 0.00001, 'batch_size': 64}
        {'z_dim': 512, 'ngf':128, 'ndf':64, 'num_epochs': num_epochs, 'lr_g': 0.0005, 'lr_d': 0.00001, 'batch_size': 64},
        {'z_dim': 768, 'ngf':256, 'ndf':128, 'num_epochs': num_epochs, 'lr_g': 0.0005, 'lr_d': 0.00001, 'batch_size': 32},
    ]

    for params in GAN_params:
        dataloader = DataLoader(data, batch_size=params['batch_size'], shuffle=True)
        print(f"Training GAN with params: {params}")
        path = os.path.join(os.getcwd(), 'models')
        model_name = f'GAN_{params["z_dim"]}_{params["ngf"]}_{params["ndf"]}'
        base_path = os.path.join(path, model_name)
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(os.path.join(base_path, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'npy'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'wav'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'png'), exist_ok=True)

        # Unpack the returned values and check if GAN training is skipped
        result = train_model('GAN', params, dataloader, base_path)
        if result[0] is None:
            print("GAN training skipped.")
            continue

        generator, discriminator, dlosses, glosses, optimizer_G, optimizer_D = result

        # Only save if the models are not None
        if discriminator is not None and generator is not None:
            torch.save(discriminator.state_dict(), base_path+'/discriminator_final.pt')
            torch.save(generator.state_dict(), base_path+'/generator_final.pt')

        plot_losses(glosses, dlosses, base_path+'/plots') # save plots

        generate_samples(generator, params['z_dim'], num_samples=25, path_npy=base_path+'/npy', path_png=base_path+'/png', path_wav=base_path+'/wav')

        optimizer_G.zero_grad(set_to_none=True)
        optimizer_D.zero_grad(set_to_none=True)
        del generator, discriminator, optimizer_G, optimizer_D
        torch.cuda.empty_cache()
        gc.collect()

    for params in VAE_params:
        dataloader = DataLoader(data, batch_size=params['batch_size'], shuffle=True)
        print(f"Training VAE with params: {params}")
        path = os.path.join(os.getcwd(), 'models')
        model_name = f'VAE_{params["z_dim"]}_{params["nef"]}_{params["ndf"]}'
        base_path = os.path.join(path, model_name)
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(os.path.join(base_path, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'npy'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'wav'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'png'), exist_ok=True)

        vae, losses, optimizer = train_model('VAE', params, dataloader, base_path)
        
        torch.save(vae.state_dict(), base_path+'/vae_final.pt')
        plot_losses(losses, [], base_path+'/plots')
        
        generate_samples(vae, params['z_dim'], num_samples=25, path_npy=base_path+'/npy', path_png=base_path+'/png', path_wav=base_path+'/wav')

        optimizer.zero_grad(set_to_none=True)
        del vae, optimizer
        torch.cuda.empty_cache()
        gc.collect()



    # for params in VAE_params:
    #     print(f"Training VAE with params: {params}")
    #     model = train_model('VAE', params, dataloader)
    #     # train model 
    #     # save model 
    #     # generate samples
    #     # save plots
    #     # compute metrics of training samples
    #     # Free GPU memory
