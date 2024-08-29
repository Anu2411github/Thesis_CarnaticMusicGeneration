import sys
sys.path.append('/home/msp/Anusha/Thesis_New/Vggish')

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.signal import stft
import os
from tqdm import tqdm
from dataset import MusicDataset
import torch
import torch.nn.functional as F
from torchvision import transforms
from pytorch_msssim import ms_ssim
import torch
import vggish_input
import vggish_params
import vggish_slim
from scipy.io import wavfile
from scipy.linalg import sqrtm

def MSE(real_samples, generated_samples):
    # for each generated sample, find miunum MSE with one real sample
    # return the average of all the minimum MSEs
    min_mses = [] 
    for gen_sample in generated_samples:
        mse_list = [np.mean((gen_sample - real.detach().cpu().numpy())**2) for real in real_samples]
        min_mses.append(min(mse_list))
    return np.mean(min_mses)

# def Spectral_Convergence(real_samples, generated_samples):
#     """
#     Compute the average Spectral Convergence between lists of real and generated log Mel spectrograms.
    
#     Args:
#     real_samples (list of np.array): List of real audio spectrograms, each of shape (200, 200).
#     generated_samples (list of np.array): List of generated audio spectrograms, each of shape (200, 200).
    
#     Returns:
#     float: The average spectral convergence metric for the best match between generated and any real samples.
#     """
#     # Store the minimum spectral convergence for each generated sample
#     min_convergences = []

#     # Iterate over each generated spectrogram
#     for generated in generated_samples:
#         # Calculate the spectral convergence between this generated and all real spectrograms
#         convergences = [np.mean(np.abs(generated - real.detach().cpu().numpy()) / (np.abs(real.detach().cpu().numpy()) + np.abs(generated) + 1e-8)) for real in real_samples]
        
#         # Find the minimum convergence for this generated spectrogram (best match scenario)
#         min_convergences.append(min(convergences))

#     # Return the average of the minimum convergences
#     return np.mean(min_convergences)

def MS_SSIM(real_samples, generated_samples):
    # Convert real_samples and generated_samples to tensors
    transform = transforms.ToTensor()
    real_tensors = [img.unsqueeze(0).unsqueeze(0) for img in real_samples]  # Convert each image to a tensor with batch dimension
    generated_tensors = [transform(img).unsqueeze(0).to('cuda') for img in generated_samples]  # Same for generated images

    ms_ssim_averages = []
    
    for gen_tensor in generated_tensors:
        ms_ssim_values = []
        
        for real_tensor in real_tensors:
            # Ensure both tensors have the same shape
            if real_tensor.shape != gen_tensor.shape:
                print(real_tensor.shape, gen_tensor.shape)
                raise ValueError("Input images must have the same dimensions")

            ms_ssim_value = ms_ssim(gen_tensor, real_tensor, data_range=2.0)
            ms_ssim_values.append(ms_ssim_value.item())
        
        average_ms_ssim = sum(ms_ssim_values) / len(ms_ssim_values)
        ms_ssim_averages.append(average_ms_ssim)
    
    return np.mean(ms_ssim_averages)
'''
#def get_vggish_embeddings(samples):
    """
    Compute VGGish embeddings for the given audio samples.
    
    Args:
    samples (list of np.array): List of audio samples. Each sample is expected to be a 1D numpy array representing raw audio.
    
    Returns:
    np.array: A numpy array of VGGish embeddings.
    """
    # Load VGGish model
    model = vggish_slim.define_vggish_slim()
    checkpoint_path = '/home/msp/Anusha/Thesis_New/Vggish/vggish_model.ckpt'
    vggish_slim.load_vggish_slim_checkpoint(model, checkpoint_path)
    model.eval()
    
    embeddings = []
    
    for sample in samples:
        # Convert sample to the expected input format for VGGish
        sample = vggish_input.waveform_to_examples(sample, vggish_params.SAMPLE_RATE)
        
        with torch.no_grad():
            # Convert to tensor
            sample_tensor = torch.from_numpy(sample).float()
            
            # Get embeddings
            embedding = model(sample_tensor)
            embeddings.append(embedding.cpu().numpy())
    
    return np.vstack(embeddings)

#def FAD(real_embeddings, generated_embeddings):
    """
    Compute the Fréchet Audio Distance (FAD) between real and generated embeddings.
    
    Args:
    real_embeddings (np.array): Array of real embeddings.
    generated_embeddings (np.array): Array of generated embeddings.
    
    Returns:
    float: The FAD score.
    """
    # Compute mean and covariance of real embeddings
    mu_real = np.mean(real_embeddings, axis=0)
    sigma_real = np.cov(real_embeddings, rowvar=False)
    
    # Compute mean and covariance of generated embeddings
    mu_generated = np.mean(generated_embeddings, axis=0)
    sigma_generated = np.cov(generated_embeddings, rowvar=False)
    
    # Compute FAD using the Fréchet distance formula
    mu_diff = mu_real - mu_generated
    covmean, _ = sqrtm(sigma_real.dot(sigma_generated), disp=False)
    
    # Check if covmean is complex due to numerical issues
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fad = np.sum(mu_diff**2) + np.trace(sigma_real + sigma_generated - 2 * covmean)
    
    return fad
'''

def evaluate_samples(real_train_samples, real_test_samples, generated_samples_list, model_names):

    """
    Evaluate the generated samples using different metrics
    inputs:
    real_train_samples: np.array of shape (num_samples, num_features)
    real_test_samples: np.array of shape (num_samples, num_features)
    generated_samples_list: list of np.arrays of shape (num_samples, num_features) [one for each model]

    returns:
    train_metrics: dict containing the metrics for the training set
    val_metrics: dict containing the metrics for the validation set
    """

    # Only use the first 10 samples from real training and validation samples
    real_train_samples = real_train_samples[:10]
    real_test_samples = real_test_samples[:10]

    #metrics = ['MSE', 'Spectral Convergance', 'MS-SSIM', 'FAD']
    metrics = ['MSE', 'MS-SSIM']
    models_metrics = {}

    for idx, generated_samples in enumerate(generated_samples_list):
        train_metrics = {}
        val_metrics = {}   

        for metric in metrics:
            if metric == 'MSE':
                train_metric = MSE(real_train_samples, generated_samples)
                val_metric = MSE(real_test_samples, generated_samples)
            # elif metric == 'Spectral Convergance':
            #     train_metric = Spectral_Convergence(real_train_samples, generated_samples)
            #     val_metric = Spectral_Convergence(real_test_samples, generated_samples)
            elif metric == 'MS-SSIM':
                train_metric = MS_SSIM(real_train_samples, generated_samples)
                val_metric = MS_SSIM(real_test_samples, generated_samples)

            # elif metric == 'FAD':
            #     #extract embeddings
            #     train_vggish_embeddings = get_vggish_embeddings(real_train_samples)
            #     val_vggish_embeddings = get_vggish_embeddings(real_test_samples)
            #     generated_vggish_embeddings = get_vggish_embeddings(generated_samples)

            # if metric == 'FAD':
            #     # Extract embeddings
            #     train_vggish_embeddings = get_vggish_embeddings(real_train_samples)
            #     val_vggish_embeddings = get_vggish_embeddings(real_test_samples)
            #     generated_vggish_embeddings = get_vggish_embeddings(generated_samples)

            #     # Compute FAD
            #     train_metric = FAD(train_vggish_embeddings, generated_vggish_embeddings)
            #     val_metric = FAD(val_vggish_embeddings, generated_vggish_embeddings)

            train_metrics[metric] = train_metric
            val_metrics[metric] = val_metric

        models_metrics[model_names[idx]] = (train_metrics, val_metrics)

        print(f"Model {model_names[idx]}")
        print(f"Train Metrics: {train_metrics}")
        print(f"Validation Metrics: {val_metrics}")

        plot_similarity(train_vggish_embeddings, val_vggish_embeddings, generated_vggish_embeddings, method='tsne')
        plot_similarity(train_vggish_embeddings, val_vggish_embeddings, generated_vggish_embeddings, method='pca')
        plot_similarity(train_vggish_embeddings, val_vggish_embeddings, generated_vggish_embeddings, method='umap')
    return train_metrics, val_metrics


def plot_similarity(real_train_samples, real_test_samples, generated_samples_list, method='tsne'):
    """
    Plot the similarity between the real and generated samples using t-SNE, PCA, or UMAP.
    
    Inputs:
    - real_train_samples: np.array of shape (num_samples, 128)
    - real_test_samples: np.array of shape (num_samples, 128)
    - generated_samples_list: list of np.arrays of shape (num_samples, 128) [one for each model]
    - method: 'tsne', 'pca', or 'umap' (default: 'tsne')
    
    Output:
    - img_path: Path to the saved image file.
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import umap
    # Combine all samples for dimensionality reduction
    all_samples = [real_train_samples, real_test_samples] + generated_samples_list
    all_samples_combined = np.vstack(all_samples)
    
    # Apply the selected dimensionality reduction method
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'tsne', 'pca', or 'umap'")
    
    reduced_samples = reducer.fit_transform(all_samples_combined)
    
    # Split the reduced samples back into their respective groups
    num_train_samples = len(real_train_samples)
    num_val_samples = len(real_test_samples)
    
    train_samples_reduced = reduced_samples[:num_train_samples]
    val_samples_reduced = reduced_samples[num_train_samples:num_train_samples + num_val_samples]
    generated_samples_reduced_list = [
        reduced_samples[num_train_samples + num_val_samples + i * num_train_samples : 
                        num_train_samples + num_val_samples + (i + 1) * num_train_samples]
        for i in range(len(generated_samples_list))
    ]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(train_samples_reduced[:, 0], train_samples_reduced[:, 1], label='Real Train', alpha=0.5)
    plt.scatter(val_samples_reduced[:, 0], val_samples_reduced[:, 1], label='Real Validation', alpha=0.5)
    
    for i, gen_samples_reduced in enumerate(generated_samples_reduced_list):
        plt.scatter(gen_samples_reduced[:, 0], gen_samples_reduced[:, 1], label=f'Generated Model {i + 1}', alpha=0.5)
    
    plt.legend()
    plt.title(f'Similarity Plot using {method.upper()}')
    img_path = f'{method}_similarity.png'
    plt.savefig(img_path)
    plt.close()
    
    return img_path


def plot_epoch_metrics(model_paths, model_names):
    """
    inputs:
    model_paths: list of paths to the model directories
    """
    plt.style.use('ggplot')
    def create_plot(data, title, ylabel, filename):
        plt.figure(figsize=(10, 6))
        
        for idx, model in enumerate(model_paths):
            plt.plot(range(5, 105, 5), data[model], label=model_names[idx], marker='o', linewidth=2)

        # Customize ticks and labels
        plt.xticks(ticks=range(5, 105, 10), fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.title(title, fontsize=16, fontweight='bold')
        
        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Improve legend placement and appearance
        plt.legend(title='Models', fontsize=12, title_fontsize='13', loc='best', frameon=True)
        
        # Tight layout to reduce white space
        plt.tight_layout()
        
        # Save the plot with a high DPI for better quality
        plt.savefig(f'{filename}', dpi=300)
        plt.close()


    assert len(model_paths) == len(model_names), "Number of model paths and model names should be the same"
    epochs_samples_paths = [[model_path + f'/epoch_{i}' for i in range(5, 105,5)] for model_path in model_paths]
    #metrics = ['MSE', 'Spectral Convergance', 'MS-SSIM']
    metrics = ['MSE', 'MS-SSIM', 'FAD']
    paths = ['/home/msp/Downloads/Indian_Raga_Dataset', '/home/msp/Anusha/Thesis/dataset/CMR_full_dataset_1.0/audio','/home/msp/Downloads/carnatic_varnam_(1)/carnatic_varnam_1.0/Audio']
    test_data_path = ['/home/msp/Anusha/Thesis/dataset/CMR_subset_1.0/CMR_subset_1.0/audio']
    data = MusicDataset(paths, segment_length=200, target_sr=16000, train=True)
    test_data = MusicDataset(test_data_path, segment_length=200, target_sr=16000, train=False)
    real_train_samples_idx = np.random.randint(0, data.data.shape[0], size=1500)
    real_test_samples_idx = np.random.randint(0, test_data.data.shape[0], size=1500)
    real_train_samples = data[real_train_samples_idx]
    real_test_samples = test_data[real_test_samples_idx]

    for metric in metrics:
        chart_data_train = {model: [] for model in model_paths}
        chart_data_val = {model: [] for model in model_paths}
        for idx, epoch_samples_path in tqdm(enumerate(epochs_samples_paths), desc=f"Computing {metric}"):
            for samples_path in tqdm(epoch_samples_path, desc=f"Computing {metric} for {model_names[idx]}"):
                samples_path = samples_path + '/npy'
                generated_samples_paths = os.listdir(samples_path)
                generated_samples = [np.load(samples_path + '/' + sample_path) for sample_path in generated_samples_paths]

                if metric == 'MSE':
                    train_metric = MSE(real_train_samples, generated_samples)
                    val_metric = MSE(real_test_samples, generated_samples)
                # elif metric == 'Spectral Convergance':
                #     train_metric = Spectral_Convergence(real_train_samples, generated_samples)
                #     val_metric = Spectral_Convergence(real_test_samples, generated_samples)
                elif metric == 'MS-SSIM':
                    train_metric = MS_SSIM(real_train_samples, generated_samples)
                    val_metric = MS_SSIM(real_test_samples, generated_samples)

                # if metric == 'FAD':
                #     # Extract embeddings
                #     train_vggish_embeddings = get_vggish_embeddings(real_train_samples)
                #     val_vggish_embeddings = get_vggish_embeddings(real_test_samples)
                #     generated_vggish_embeddings = get_vggish_embeddings(generated_samples)

                #     # Compute FAD
                #     train_metric = FAD(train_vggish_embeddings, generated_vggish_embeddings)
                #     val_metric = FAD(val_vggish_embeddings, generated_vggish_embeddings)

                chart_data_train[model_paths[idx]].append(train_metric)
                chart_data_val[model_paths[idx]].append(val_metric)

        # Plot for Train data
        # Plot for Training data
        create_plot(chart_data_train, f'{metric} - Train Data', metric, f'{metric}_train_metric.png')

        # Plot for Validation data
        create_plot(chart_data_val, f'{metric} - Test Data', metric, f'{metric}_test_metric.png')
        # Function to customize and save the plot

            
            
if __name__ == '__main__':

    # evaluate_samples function - will give test and train results using final model (will also make tsne, pca, umap plots)

    # plot_epoch_metrics function - will give the metrics for each model at different epochs (will also make plots for each metric)

    #This is for DCGAN
    model_paths = ['/home/msp/Anusha/Thesis_New/models/GAN_256_64_64', '/home/msp/Anusha/Thesis_New/models/GAN_512_128_64','/home/msp/Anusha/Thesis_New/models/GAN_768_256_128']
    model_names = ['DCGAN_001', 'DCGAN_002', 'DCGAN_003']
    
    # Uncomment this to run for VAE
    # model_paths = ['/home/msp/Anusha/Thesis_New/models/VAE_256_64_64', '/home/msp/Anusha/Thesis_New/models/VAE_512_64_128','/home/msp/Anusha/Thesis_New/models/VAE_768_128_256']
    # model_names = ['VAE_256_64_64', 'VAE_512_64_128', 'VAE_768_128_256']
    plot_epoch_metrics(model_paths, model_names)
