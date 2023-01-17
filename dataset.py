import os
import numpy as np
import torchaudio

class Dataset():
    def __init__(self,):

        self.noisy_path = 'vbd_path/noisy_testset_wav_16k'
        self.clean_path = 'vbd_path/clean_testset_wav_16k'
        self.ref_path = 'vbd_path/ref_test'

        self.data_list = os.listdir(self.noisy_path)

        self.ref_length = 16384 * 15

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        noisy, _ = torchaudio.load('{}/{}'.format(self.noisy_path, item))
        clean, _ = torchaudio.load('{}/{}'.format(self.clean_path, item))
        ref = np.load('{}/{}_ref.npy'.format(self.ref_path, item.split('.')[0].split('/')[-1]), allow_pickle=True).item()
        ref_stft = ref['stfts']
        ref_mfcc = ref['mfccs']
        noisy, clean = noisy.squeeze(), clean.squeeze()
        return noisy, clean, ref_stft, ref_mfcc
