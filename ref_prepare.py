
import os
import random
import numpy as np
import torch
import copy
import torchaudio
from tqdm import tqdm
from utils import stft as stft_f

noisy_path = 'vbd_path/noisy_testset_wav_16k'
clean_path = 'vbd_path/clean_testset_wav_16k'
ref_path = 'vbd_path/ref_test'

mfcc_f = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=13, dct_type=2, norm='ortho', log_mels=False,
                                       melkwargs={'n_fft': 512, 'win_length': 512, 'hop_length': 128, 'pad': 0,
                                                  'n_mels': 128, 'power': 1, 'window_fn': torch.hann_window})

test_list = []
test_dict = {}

ref_length = 16384 * 15

for item in os.listdir(noisy_path):
    test_list.append(['{}/{}'.format(noisy_path, item), '{}/{}'.format(clean_path, item)])

    speaker = item.split('_')[0]

    if speaker in test_dict.keys():
        test_dict[speaker].append(['{}/{}'.format(noisy_path, item), '{}/{}'.format(clean_path, item)])
    else:
        test_dict[speaker] = [['{}/{}'.format(noisy_path, item), '{}/{}'.format(clean_path, item)]]

for data in tqdm(test_list):
    _, path = data

    speaker = data[0].split('/')[-1].split('_')[0]
    org = copy.deepcopy(test_dict[speaker])
    org.remove(data)
    random.shuffle(org)

    waves, mfccs, stfts = None, None, None
    while waves is None or waves.shape[1] < 1 * ref_length:
        _, path = org.pop()
        wave = torchaudio.load(path)[0]
        mfcc = mfcc_f(wave).squeeze()
        stft = stft_f(wave).squeeze()

        if waves is None:
            waves, mfccs, stfts = wave, mfcc, stft
        else:
            waves = torch.cat([waves, wave], dim=1)
            mfccs = torch.cat([mfccs, mfcc], dim=1)
            stfts = torch.cat([stfts, stft], dim=1)

    stfts = stfts[:, :ref_length // 128 + 1, :]
    mfccs = mfccs[:, :ref_length // 128 + 1]
    waves = waves[:, :ref_length]

    ref = {
        'stfts': stfts,
        'mfccs': mfccs,
    }

    name = data[0].split('/')[-1].split('.')[0] + '_ref.npy'
    np.save('{}/{}'.format(ref_path, name), ref)