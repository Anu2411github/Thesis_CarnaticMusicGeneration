# Thesis Project: Generative Approaches to High-Fidelity Indian Classical Music Generation

This repository contains the code and data used in my thesis project, where I explore generative models Deep Convolutional Generative Adversarial Networks (DCGAN) and Variational Autoencoders (VAE) for generating Carnatic Indian Classical Music.

# Introduction
This project focuses on applying generative models to music spectrograms, with the goal of synthesizing new Music audio data. The models are trained on spectrograms extracted from 3 different music datasets, and their performance is evaluated using metrics MSE, MS-SSIM.

# Deep Concolutional Generative Adversarial Network (DCGAN)
The GAN model includes:

Generator: Generates fake spectrograms from random noise.
Discriminator: Distinguishes between real and fake spectrograms.
The GAN is trained using adversarial loss, where the generator tries to fool the discriminator.

# Variational Autoencoder (VAE)
The VAE model consists of an encoder and a decoder:

Encoder: Encodes the input spectrogram into a latent space representation.
Decoder: Reconstructs the spectrogram from the latent space.
The VAE is trained using a combination of reconstruction loss (MSE/BCE) and Kullback-Leibler divergence (KLD) loss.

# Dataset
The dataset consists of music audio files, which are processed into Mel spectrograms. The spectrograms are segmented into smaller frames for training. The dataset is split into training and testing sets.

Sample Rate: 16 kHz
Segment Length: 200 frames

# Training

DCGAN Training
The DCGAN is trained over 100 epochs with the following parameters:

Latent Dimensions: 256, 512, 768
Learning Rates: 0.0005 for Generator, 0.00001 for Discriminator
Batch Size: 32-64
The losses for the generator and discriminator are tracked and plotted.

VAE Training
The VAE is trained over 100 epochs with the following parameters:

Latent Dimensions: 256, 512, 768
Learning Rate: 1e-5
Batch Size: 32-64

The loss function includes MSE/BCE and KLD, with a gradually increasing beta coefficient.

# Evaluation
The models are evaluated using the following metrics:

MSE: Mean Squared Error between the real and generated spectrograms.
MS-SSIM: Multi-Scale Structural Similarity Index between real and generated spectrograms.

# Results
The results include:

Loss plots for both DCGAN and VAE models.
Generated spectrograms and audio files.
Metric evaluations on test datasets.

# Dependencies
The dependencies include:

PyTorch
NumPy
Librosa
Matplotlib
Scikit-learn
tqdm

# How to Run
Training: Use the following command to train the models:
python train.py --model_type [VAE/GAN] --config config.json

Evaluation: After training, evaluate the models:
python evaluate.py --model_path path_to_saved_model

Generate Samples: To generate new samples:
python generate_samples.py --model_path path_to_saved_generator

# Contact
For any questions or collaborations, please contact me at
# sanusha197@gmail.com
