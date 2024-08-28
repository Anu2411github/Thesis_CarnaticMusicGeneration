from dataset import MusicDataset
from GAN import init_GAN
from VAE import VAE
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import os 
from plots import plot_losses
from generate_samples import generate_samples
import numpy as np
import random
import gc

from tqdm import tqdm

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def vae_loss(reconstructed, original, mu, logvar, beta):
    # Calculate MSE Loss
    mse_loss = nn.functional.mse_loss(reconstructed, original, reduction='mean')
    
    # Ensure the reconstructed values are in [0, 1] for BCE calculation
    reconstructed_sigmoid = torch.sigmoid(reconstructed)
    original_sigmoid = torch.sigmoid(original)
    # Calculate Binary Cross-Entropy Loss
    bce_loss = nn.functional.binary_cross_entropy(reconstructed_sigmoid, original_sigmoid, reduction='mean')

    # Calculate KL Divergence Loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld_loss /= original.size(0)  # Normalize by batch size

    # Combine losses with a weighting for KL divergence
    total_loss = mse_loss + (beta * kld_loss)

    return total_loss, (mse_loss, bce_loss, kld_loss)


def train_VAE(vae, num_epochs, dataloader, optimizer, model_path):
    all_mse_losses = []
    all_kld_losses = []
    all_bce_losses = []
    all_losses = []
    beta = 0
    beta_increment = 0.015

    # Outer tqdm for epochs
    epoch_progress = tqdm(range(num_epochs), desc='Epochs')
    for epoch in epoch_progress:
        mse_losses = []
        kld_losses = []
        bce_losses = []
        losses = []
        # Inner tqdm for steps within each epoch
        step_progress = tqdm(enumerate(dataloader), desc='Steps', leave=False, total=len(dataloader))
        for i, (mel_spectrograms) in step_progress:
            mel_spectrograms = mel_spectrograms.to('cuda').unsqueeze(1)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(mel_spectrograms)
            loss, (mse_loss, bce_loss, kld_loss) = vae_loss(recon_batch, mel_spectrograms, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            mse_losses.append(mse_loss.item())
            kld_losses.append(kld_loss.item())
            bce_losses.append(bce_loss.item())
            losses.append(loss.item())

        all_mse_losses.extend(mse_losses)
        all_kld_losses.extend(kld_losses)
        all_bce_losses.extend(bce_losses)
        all_losses.extend(losses)


        # Gradually increase KL weight
        beta = min(1.0, beta + beta_increment)
        avg_mse_loss = sum(mse_losses) / len(mse_losses)
        avg_kld_loss = sum(kld_losses) / len(kld_losses)
        avg_bce_loss = sum(bce_losses) / len(bce_losses)
        avg_loss = sum(all_losses) / len(all_losses)

        print(f'Epoch {epoch + 1} - MSE Loss: {avg_mse_loss:.4f}, KLD Loss: {avg_kld_loss:.4f}, BCE Loss: {avg_bce_loss:.4f}, Total Loss: {avg_loss:.4f} Beta: {beta:.4f}')

        #epoch_progress.set_postfix_str(f'Loss: {sum(losses)/len(losses):.4f}')

        if (epoch + 1) % 5 == 0:
            path = model_path + f'/epoch_{epoch + 1}'
            os.makedirs(path, exist_ok=True)
            os.makedirs(path+'/npy', exist_ok=True)
            os.makedirs(path+'/png', exist_ok=True)
            os.makedirs(path+'/wav', exist_ok=True)
            generate_samples(vae.decoder, vae.latent_dim, num_samples=50, path_npy=path+'/npy', path_png=path+'/png', path_wav=path+'/wav',vae=True)
        if (epoch + 1) % 10 == 0:
            torch.save(vae.state_dict(), os.path.join(model_path, f'vae_epoch_{epoch + 1}.pt'))
    # save all_losses
    np.save(os.path.join(model_path, 'all_mse_losses.npy'), np.array(all_mse_losses))
    np.save(os.path.join(model_path, 'all_kld_losses.npy'), np.array(all_kld_losses))
    np.save(os.path.join(model_path, 'all_bce_losses.npy'), np.array(all_bce_losses))
    np.save(os.path.join(model_path, 'all_losses.npy'), np.array(all_losses))

    return vae, all_mse_losses, all_kld_losses

def train_model(model_type, params, dataloader,model_path):
    if model_type == 'VAE':
        # initialize VAE
        vae = VAE(latent_dim=params['z_dim'], nef=params['nef'], ndf=params['ndf'], nc=1)
        vae = vae.to('cuda')
        optimizer = torch.optim.Adam(vae.parameters(), lr=params['lr'])
        vae, all_mse_losses, all_kld_losses = train_VAE(vae, params['num_epochs'], dataloader, optimizer, model_path)
        return vae, all_mse_losses, all_kld_losses, optimizer
    elif model_type == 'GAN':
        G, D = init_GAN(ngpu=1, nz=params['z_dim'], ngf=params['ngf'], nc=1, ndf=params['ndf'])
        G = G.to('cuda')
        D = D.to('cuda')
        criterion = nn.BCELoss()
        optimizer_G = torch.optim.Adam(G.parameters(), lr=params['lr_g'], betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(D.parameters(), lr=params['lr_d'], betas=(0.5, 0.999))
        G, D, dlosses, glosses = train_GAN(G, D, params['num_epochs'], dataloader, optimizer_D, optimizer_G, criterion, params['z_dim'], model_path)
        return G, D, dlosses, glosses, optimizer_G, optimizer_D
    
    else:
        raise ValueError(f"Model type {model_type} not recognized")


if __name__ == '__main__':
    set_seed(24) 

    paths = ['/home/msp/Downloads/Indian_Raga_Navee_Dataset', '/home/msp/Anusha/Thesis/dataset/CMR_full_dataset_1.0/audio','/home/msp/Downloads/carnatic_varnam_(1)/carnatic_varnam_1.0/Audio']
    data = MusicDataset(paths, segment_length=200, target_sr=16000)
    num_epochs = 100
    VAE_params = [
        # {'z_dim': 256, 'nef':64 , 'ndf':64, 'num_epochs': num_epochs, 'lr': 1e-5, 'batch_size': 4},
        {'z_dim': 512, 'nef':64 , 'ndf':128, 'num_epochs': num_epochs, 'lr': 1e-5, 'batch_size': 4},
        {'z_dim': 768, 'nef':128 , 'ndf':256, 'num_epochs': num_epochs, 'lr': 1e-5, 'batch_size': 4},
    ]

    GAN_params = [
        {'z_dim': 256, 'ngf':64, 'ndf':64, 'num_epochs': num_epochs, 'lr_g': 0.0005, 'lr_d': 0.00001, 'batch_size': 64},
        {'z_dim': 512, 'ngf':128, 'ndf':64, 'num_epochs': num_epochs, 'lr_g': 0.0005, 'lr_d': 0.00001, 'batch_size': 64},
        {'z_dim': 768, 'ngf':256, 'ndf':128, 'num_epochs': num_epochs, 'lr_g': 0.0005, 'lr_d': 0.00001, 'batch_size': 32},
    ]

    
    # for params in GAN_params:
    #     dataloader = DataLoader(data, batch_size=params['batch_size'], shuffle=True)
    #     print(f"Training GAN with params: {params}")
    #     path = os.path.join(os.getcwd(), 'models')
    #     model_name = f'GAN_{params["z_dim"]}_{params["ngf"]}_{params["ndf"]}'
    #     base_path = os.path.join(path, model_name)
    #     os.makedirs(base_path, exist_ok=True)
    #     os.makedirs(os.path.join(base_path, 'plots'), exist_ok=True)
    #     os.makedirs(os.path.join(base_path, 'npy'), exist_ok=True)
    #     os.makedirs(os.path.join(base_path, 'wav'), exist_ok=True)
    #     os.makedirs(os.path.join(base_path, 'png'), exist_ok=True)

    #     generator, discriminator, dlosses, glosses, optimizer_G, optimizer_D = train_model('GAN', params, dataloader, base_path)

    #     torch.save(discriminator.state_dict(), base_path+'/discriminator_final.pt')
    #     torch.save(generator.state_dict(), base_path+'/generator_final.pt')

    #     plot_losses(glosses,dlosses,base_path+'/plots') # save plots

    #     # generate and save samples (100 npy, 1 png of 100 samples, 100 wav)
    #     generate_samples(generator, params['z_dim'], num_samples=25, path_npy=base_path+'/npy', path_png=base_path+'/png', path_wav=base_path+'/wav')

    #     # Clear gradients
    #     optimizer_G.zero_grad(set_to_none=True)
    #     optimizer_D.zero_grad(set_to_none=True)
    #     # Free GPU memory
    #     del generator, discriminator, optimizer_G, optimizer_D
    #     torch.cuda.empty_cache()  # Clear cache
    #     gc.collect()  # Clean up Python's garbage collection


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

        vae, all_mse_losses, all_kld_losses, optimizer = train_model('VAE', params, dataloader, base_path)
        
        torch.save(vae.state_dict(), base_path+'/vae_final.pt')
        plot_losses(all_mse_losses, all_kld_losses, base_path+'/plots', vae=True) 
        
        generate_samples(vae.decoder, params['z_dim'], num_samples=25, path_npy=base_path+'/npy', path_png=base_path+'/png', path_wav=base_path+'/wav', vae=True)

        optimizer.zero_grad(set_to_none=True)
        del vae, optimizer
        torch.cuda.empty_cache()
        gc.collect()


