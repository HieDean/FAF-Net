import torch
import torchaudio
import torch.nn.functional as F


mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=13, dct_type=2, norm='ortho', log_mels=False,
                                       melkwargs={'n_fft': 512, 'win_length': 512, 'hop_length': 128, 'pad': 0,
                                                  'n_mels': 128, 'power': 1, 'window_fn': torch.hann_window}).cuda()


def stft(wave):
    return torch.stft(wave, n_fft=512, hop_length=128, win_length=512, window=torch.hann_window(512).to(wave.device))


def istft(spectra, length):
    return torch.istft(spectra, length=length, n_fft=512, hop_length=128, win_length=512, window=torch.hann_window(512).to(spectra.device))

def batch_match(noisy, mfccs):
    b, _ = noisy.shape
    b, feat_dim, l_ = mfccs.shape

    k = mfccs.reshape(b, 1, feat_dim, l_)
    k = F.unfold(k, kernel_size=(13, 3), padding=(0, 1))
    k = F.normalize(k, dim=1)

    feature = mfcc(noisy)
    l = feature.shape[2]
    feature = F.unfold(feature.reshape(b, 1, feat_dim, l), kernel_size=(13, 3), padding=(0, 1))
    query = feature.permute(0, 2, 1)
    query = F.normalize(query, dim=2)
    similarity = torch.bmm(query, k)
    similarity = torch.topk(similarity, k=5)[1]
    return similarity