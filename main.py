import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from utils import *

# import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')

    # general options
    parser.add_argument('--dataset', type=str, default=None, help='Dataset path.')
    args = parser.parse_args()
    print(args)

    args.dataset = 'C:\\Users\\keill\\Desktop\\Datasets\\Buzz\\Training Data\\Flower Buzzes\\'

    y, sr = librosa.load(os.path.join(args.dataset, '1_12_cit1.wav'))
    print(y, y.shape, sr)
    # trim silent edges
    buzz_sound, _ = librosa.effects.trim(y)
    # librosa.display.waveplot(buzz_sound, sr=sr)
    print(buzz_sound)

    n_fft = 2048
    hop_length = 512
    D = np.abs(librosa.stft(buzz_sound, n_fft=n_fft, hop_length=hop_length))
    # print(D, D.shape)
    # librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()
    # plt.savefig(plot + "/" + name + "_plot_static_conv.png")

    # DB = librosa.amplitude_to_db(D, ref=np.max)
    # librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    # plt.colorbar(format='%+2.0f dB')
    # plt.tight_layout()
    # plt.show()

    n_mels = 128
    S = librosa.feature.melspectrogram(buzz_sound, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
