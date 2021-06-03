import os
import argparse
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

import torch


def str2bool(v):
    """
    Function to transform strings into booleans.

    v: string variable
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_best_models(net, optimizer, output_path, best_records, epoch, acc, acc_cls, cm):
    if len(best_records) < 5:
        best_records.append({'epoch': epoch, 'acc': acc, 'acc_cls': acc_cls, 'cm': cm})

        torch.save(net.state_dict(), os.path.join(output_path, 'model_' + str(epoch) + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(output_path, 'opt_' + str(epoch) + '.pth'))
    else:
        # find min saved acc
        min_index = 0
        for i, r in enumerate(best_records):
            if best_records[min_index]['acc_cls'] > best_records[i]['acc_cls']:
                min_index = i
        # check if currect acc is greater than min saved acc
        if acc_cls > best_records[min_index]['acc_cls']:
            # if it is, delete previous files
            min_step = str(best_records[min_index]['epoch'])

            os.remove(os.path.join(output_path, 'model_' + min_step + '.pth'))
            os.remove(os.path.join(output_path, 'opt_' + min_step + '.pth'))

            # replace min value with current
            best_records[min_index] = {'epoch': epoch, 'acc': acc, 'acc_cls': acc_cls, 'cm': cm}

            # save current model
            torch.save(net.state_dict(), os.path.join(output_path, 'model_' + str(epoch) + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(output_path, 'opt_' + str(epoch) + '.pth'))


# reference: https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
def plot_sound(dataset):

    # y = audio time series, sr = sampling rate of y
    y, sr = librosa.load(os.path.join(dataset, '1_12_cit1.wav'))
    print(y.shape, sr, librosa.get_duration(y))
    # trim silent edges
    buzz_sound, _ = librosa.effects.trim(y)
    # librosa.display.waveplot(buzz_sound, sr=sr)
    print(buzz_sound)

    n_fft = 2048
    hop_length = 512
    # D = np.abs(librosa.stft(buzz_sound, n_fft=n_fft, hop_length=hop_length))
    # print(D, D.shape)
    # librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()

    # DB = librosa.amplitude_to_db(D, ref=np.max)
    # librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    # plt.colorbar(format='%+2.0f dB')
    # plt.tight_layout()
    # plt.show()

    n_mels = 128
    S = librosa.feature.melspectrogram(buzz_sound, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    print('S.shape', S.shape)
    S_DB = librosa.power_to_db(S, ref=np.max)
    print('S_DB.shape', S_DB.shape)
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    # plt.savefig("C:\\Users\\keill\\Desktop\\plot.png")
    plt.show()