import os
import argparse
# import librosa
# import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# import torch


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


def union_segments(non_background_preds, non_background_files, non_background_seconds):
    cur_pred = non_background_preds[0]
    cur_f = non_background_files[0]
    cur_s = non_background_seconds[0]

    union_non_background_pred = []
    union_non_background_files = []
    union_non_background_seconds = []

    for i in range(1, len(non_background_seconds)):
        if cur_pred == non_background_preds[i] and cur_f == non_background_files[i]:
            if cur_s[1] >= non_background_seconds[i][0]:
                cur_s[1] = non_background_seconds[i][1]
            else:
                union_non_background_pred.append(cur_pred)
                union_non_background_files.append(cur_f)
                union_non_background_seconds.append(cur_s)
                cur_s = non_background_seconds[i]
        else:
            union_non_background_pred.append(cur_pred)
            union_non_background_files.append(cur_f)
            union_non_background_seconds.append(cur_s)
            cur_pred = non_background_preds[i]
            cur_f = non_background_files[i]
            cur_s = non_background_seconds[i]

    # last step
    union_non_background_pred.append(cur_pred)
    union_non_background_files.append(cur_f)
    union_non_background_seconds.append(cur_s)

    return np.asarray(union_non_background_pred), np.asarray(union_non_background_files), \
           np.asarray(union_non_background_seconds)


def save_audio_file(preds, files, seconds, save_file):
    print(preds.shape, files.shape, seconds[:, 0].shape, seconds[:, 1].shape)
    array = sorted(list(zip(files, preds, seconds[:, 0], seconds[:, 1])), key=lambda x: x[0])

    formatted = []
    for i in range(len(array)):
        if array[i][1] == 1:
            p = 'flight'
        else:
            p = 'flower'
        formatted.append([array[i][0], p, "{:.3f}".format(array[i][2]), "{:.3f}".format(array[i][3])])
    # print(np.asarray(formatted))
    np.savetxt(save_file, np.asarray(formatted), fmt='%s   %s  %s   %s', delimiter='\t', newline='\n',
               header='Observation  Type    Start	Stop')


def overlap(min1, max1, min2, max2):
    return max(0.0, min(max1, max2) - max(min1, min2))


def long_records_metric(annotations, predictions):
    correct = 0
    incorrect = 0
    not_found = 0

    for i in range(len(annotations)):
        obs, cl, start, stop = annotations[i][0], annotations[i][1], annotations[i][2], annotations[i][3]
        has_overlap = False
        for j in range(len(predictions)):
            p_obs, p_cl, p_start, p_stop = predictions[j][0], predictions[j][1], predictions[j][2], predictions[j][3]
            if obs == p_obs:
                if overlap(start, stop, p_start, p_stop) > 0.0:
                    has_overlap = True
                    if cl == p_cl:
                        correct += 1
                    else:
                        incorrect += 1
                    break
        if has_overlap is False:
            not_found += 1
    print(correct, incorrect, not_found)


def read_annotations(path):
    _data = []
    for lbl in np.genfromtxt(path, dtype=None, skip_header=1):
        if lbl[1] == b'flight':
            lbl[1] = 1
        else:
            lbl[1] = 2
        _data.append([lbl[0], int(lbl[1]), lbl[2], lbl[3]])
    # print(_data)
    return np.asarray(_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--annotation', type=str, required=True, help='Path to the annotations')
    parser.add_argument('--prediction', type=str, required=True, help='Path to predictions')
    args = parser.parse_args()
    print(args)

    _labels = read_annotations(args.annotation)
    _preds = read_annotations(args.prediction)
    long_records_metric(_labels, _preds)
