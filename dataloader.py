import os
import librosa
import numpy as np

from sklearn import preprocessing

import torch
from torch.utils import data
import torchvision.transforms as transforms


class BuzzDataLoader(data.Dataset):

    def __init__(self, mode, dataset_input_path, n_fft=1024, hop_length=256, n_mels=128):
        super().__init__()

        self.mode = mode
        self.dataset_input_path = dataset_input_path
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.data, self.labels = self.make_dataset(dataset_input_path)
        self.num_classes = len(np.unique(self.labels))
        print(self.num_classes)

        if len(self.data) == 0:
            raise RuntimeError('Found 0 samples, please check the data set path')

    def make_dataset(self, path):
        assert self.mode in ['Train', 'Validation']

        if self.mode is 'Train':
            path = os.path.join(path, 'Training_Data')
        elif self.mode is 'Validation':
            path = os.path.join(path, 'Validation')

        _files = []
        _labels = []
        subfolders = os.listdir(path)
        for subf in subfolders:
            files = os.listdir(os.path.join(path, subf))  # read files of each subfolder
            for f in files:
                _files.append(os.path.join(path, subf, f))
                _labels.append(subf)

        le = preprocessing.LabelEncoder()  # 'Background_Clips': 0, 'Flight_Buzzes': 1, 'Flower_Buzzes': 2
        _labels = le.fit_transform(_labels)
        print(len(_files), len(_labels), dict(zip(le.classes_, le.transform(le.classes_))))

        return _files, _labels

    def __getitem__(self, index):
        y, sr = librosa.load(self.data[index])
        cl = self.labels[index]

        # print(librosa.get_duration(y, sr))

        buzz_sound, _ = librosa.effects.trim(y)

        mel_spectra = librosa.feature.melspectrogram(buzz_sound, sr=sr, n_fft=self.n_fft,
                                                     hop_length=self.hop_length, n_mels=self.n_mels)
        # mel_spectra = np.expand_dims(mel_spectra, axis=0)
        if len(mel_spectra.shape) == 2:
            mel_spectra = np.stack([mel_spectra] * 3, 2)  # .transpose(2, 0, 1)

        mel_spectra = (mel_spectra - np.min(mel_spectra))/np.ptp(mel_spectra)  # normalize 0 and 1

        # print(mel_spectra.shape, type(mel_spectra), np.min(mel_spectra), np.max(mel_spectra))
        # print('mel_spectra.shape', mel_spectra.shape)
        # mel_spectra_DB = librosa.power_to_db(mel_spectra, ref=np.max)

        transform = transforms.Compose([
            # transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        mel_spectra = transform(mel_spectra)

        # Returning to iterator.
        return mel_spectra.float(), cl
        # return torch.from_numpy(mel_spectra).float(), cl

    def __len__(self):
        return len(self.data)
