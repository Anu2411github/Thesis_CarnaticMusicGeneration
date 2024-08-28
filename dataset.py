from torch.utils.data import DataLoader, Dataset
import librosa
import numpy as np
import os
import torch
from tqdm import tqdm
import pickle

class MusicDataset(Dataset):
    def __init__(self, data_dir, target_sr=16000, segment_length=200, train=True):
        self.data_dir = data_dir
        self.target_sr = target_sr
        self.segment_length = segment_length
        self.data_path = '/home/msp/Anusha/Thesis_New/data'
        self.train = train
        self.data = self.load_data()


    def load_data(self):
        features = []
        if self.train:
            self.data_path = self.data_path + '/train'
        else:
            self.data_path = self.data_path + '/test'

        if not os.path.exists(self.data_path + '/' f"{self.segment_length}" + '_'+ f"{self.target_sr}"):
            os.mkdir(self.data_path + '/' f"{self.segment_length}" + '_'+ f"{self.target_sr}")
        for data_d in self.data_dir:
            files = os.listdir(data_d)
            for i, file in tqdm(enumerate(files), desc=f"Reading files from {data_d.split('/')[-1]}", total=len(files)): 
                file_path = os.path.join(data_d, file)
                if os.path.exists(self.data_path + '/' + f"{self.segment_length}" + '_'+ f"{self.target_sr}" + '/' + file.split('.')[0] + '.pkl'):
                    mel_segments = pickle.load(open(self.data_path + '/' + f"{self.segment_length}" + '_'+ f"{self.target_sr}" + '/' + file.split('.')[0] + '.pkl', 'rb'))
                else:
                    mel_segments = self.extract_features(file_path, i)
                    pickle.dump(mel_segments, open(self.data_path + '/' + f"{self.segment_length}" + '_'+ f"{self.target_sr}" + '/' + file.split('.')[0] + '.pkl', 'wb'))
                features.extend(mel_segments)

        # Stack all features and normalize globally
        features = np.array(features)
        features = (features - np.mean(features)) / np.std(features)  # Zero-centering

        # Min-max scaling to [-1, 1]
        features = 2 * (features - np.min(features)) / (np.max(features) - np.min(features)) - 1

        return features

    def extract_features(self, file_path, index):
        y, original_sr = librosa.load(file_path)

        # Resample if necessary
        if original_sr != self.target_sr:
            y = librosa.resample(y, orig_sr=original_sr, target_sr=self.target_sr)
            sr = self.target_sr
        else:
            sr = original_sr

        # 25 ms window and 10 ms stride
        n_fft = 400
        # hop_length = int(n_fft/4)
        hop_length = 160
        
        # Generate mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=200, n_fft=n_fft, hop_length=hop_length)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram)

        # Split into fixed-length segments
        segment_length = self.segment_length
        segments = []
        total_frames = mel_spectrogram.shape[1]
        for start_idx in range(0, total_frames, segment_length):
            end_idx = start_idx + segment_length
            segment = mel_spectrogram[:, start_idx:end_idx]
            if segment.shape[1] < segment_length:
                continue
            assert segment.shape[1] == segment_length, f"Wrong segment shape {segment.shape}"
            segments.append(segment)

        return segments

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel_spectrogram = torch.tensor(self.data[idx], dtype=torch.float32).to('cuda')
        return mel_spectrogram