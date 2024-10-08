import matplotlib.pyplot as plt

def plot_losses(glosses, dlosses, path=None, vae=False):
    if vae:
        plt.figure(figsize=(12, 6))
        mse_loss = glosses
        kl_loss = dlosses
        plt.plot(mse_loss, label='MSE', color='blue', linewidth=2, linestyle='--')
        plt.plot(kl_loss, label='KLD', color='red', linewidth=2, linestyle='-')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xlabel('Steps', fontsize=14, labelpad=10)
        plt.ylabel('Loss', fontsize=14, labelpad=10)
        plt.title('VAE Losses Over Epochs', fontsize=16, pad=15)
        plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
        plt.tight_layout()

        plt.ylim(0, 0.5)
        file_path = path + '/losses.png'
        plt.savefig(file_path)
        return file_path

    else:
        plt.figure(figsize=(12, 6))

        plt.plot(glosses, label='Generator Loss', color='blue', linewidth=2, linestyle='--')
        plt.plot(dlosses, label='Discriminator Loss', color='red', linewidth=2, linestyle='-')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xlabel('Steps', fontsize=14, labelpad=10)
        plt.ylabel('Loss', fontsize=14, labelpad=10)
        plt.title('GAN Losses Over Epochs', fontsize=16, pad=15)
        plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
        plt.tight_layout()
        
        file_path = path + '/losses.png'
        plt.savefig(file_path)
        return file_path
