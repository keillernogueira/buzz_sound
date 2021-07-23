import os
import math
import librosa
import numpy as np

from torch.utils import data
import torchvision.transforms as transforms


class BuzzDataLoaderAudio(data.Dataset):

    # 5512.5 == 0.25 seconds
    # 11025 == 0.5 seconds
    def __init__(self, mode, dataset_input_path, n_fft=1024, hop_length=256, n_mels=128, window=11025):
        super().__init__()

        self.mode = mode  # not used now
        self.dataset_input_path = dataset_input_path
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.window = window

        self.num_audios = 0
        self.data, self.labels = self.make_dataset(dataset_input_path)
        # self.num_classes = len(np.unique(self.labels))
        # print(self.num_classes)

        if len(self.data) == 0:
            raise RuntimeError('Found 0 samples, please check the data set path')

    def make_dataset(self, path):
        # assert self.mode in ['Train', 'Validation']

        _data = []
        _labels = []
        files = os.listdir(path)  # read files
        print(files)
        self.num_audios = len(files)
        for f in files:
            if os.path.isfile(os.path.join(path, f)):
                y, sr = librosa.load(os.path.join(path, f))
                # divide by 2 so we can have some overlapping
                _data += [f + '-' + str(s) for s in np.arange(0.0, y.shape[0]-(self.window/2.0), (self.window/2.0))]
                # print(librosa.get_duration(y, sr), y.shape,
                # np.arange(0.0, librosa.get_duration(y, sr)-self.window, self.window))
                for lbl in np.genfromtxt(os.path.join(path, 'labels', f.replace('mp3', 'txt')), dtype=None):
                    if lbl[0] == b'flight':
                        lbl[0] = 1
                    else:
                        lbl[0] = 2
                    _labels += [f + '-' + lbl[0] + "_" + lbl[1] + "_" + lbl[2]]

        print(_data[0:100], len(_data))
        print(_labels, len(_labels))
        return _data, _labels

    def __getitem__(self, index):
        f, i = self.data[index].split('-')
        y, sr = librosa.load(os.path.join(self.dataset_input_path, f))
        buzz_sound, _ = librosa.effects.trim(y)
        buzz_sound = buzz_sound[int(math.ceil(float(i))):min(int(math.ceil(float(i)))+self.window, buzz_sound.shape[0])]
        # print(f, i, y.shape, buzz_sound.shape, librosa.get_duration(y, sr), librosa.get_duration(buzz_sound, sr))

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
        return mel_spectra.float(), int(float(i))

    def __len__(self):
        return len(self.data)
