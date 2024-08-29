import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

def generate_samples(generator, z_dim, num_samples=1, path_npy=None, path_png=None, path_wav=None, vae=False):
    assert path_npy and path_png and path_wav, "Please provide paths for saving the generated spectrograms, plots, and audio files"
    generator.eval()
    z = torch.randn(num_samples, z_dim, 1, 1, device='cuda') if not vae else torch.randn(num_samples, z_dim, device='cuda')
    with torch.no_grad():
        generated_data = generator(z)
    generated_data = generated_data.squeeze(1).cpu().numpy()

    spectrogram_paths = []
    for i, spectrogram in enumerate(generated_data):
        file_path = os.path.join(path_npy, f'generated_spectrogram_{i}.npy')
        np.save(file_path, spectrogram)
        spectrogram_paths.append(file_path)

    num_rows = (num_samples + 2) // 3

    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for i, np_mel in enumerate(generated_data):
        ax = axes[i]
        ax.imshow(np_mel)
        ax.set_title(f'Spectrogram {i}')
        ax.axis('off')
    
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(path_png, 'generated_spectrogram.png'))

    sr = 16000
    n_fft = 400
    hop_length = 160

    for i in range(generated_data.shape[0]):
        generated_audio = librosa.feature.inverse.mel_to_audio(generated_data[i], sr=sr, n_fft=n_fft, hop_length=hop_length)
        file_path = os.path.join(path_wav, f'generated_audio_{i}.wav')
        sf.write(file_path, generated_audio, sr)
        # Specify your local directory to save the generated spectrograms


