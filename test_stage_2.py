import os
import torch
import numpy as np
from tqdm import tqdm
from pesq import pesq
from torch.utils.data import DataLoader

from model import ComplexUnet, ComplexUnet_
from utils import batch_match, stft, istft, mfcc
from dataset import Dataset

model_1 = ComplexUnet().cuda()
checkpoint = torch.load('ckpt_1.pth', map_location='cpu')
model_1.load_state_dict(checkpoint)
model_1.eval()

model_2 = ComplexUnet_().cuda()
checkpoint = torch.load('ckpt_2.pth', map_location='cpu')
model_2.load_state_dict(checkpoint)
model_2.eval()

dataloader = DataLoader(Dataset(), batch_size=1, num_workers=4, shuffle=False)

pesq_wb = []
for idy, item in enumerate(tqdm(dataloader)):
    noisy, clean, ref_stft, ref_mfcc = item

    noisy, ref_stft, ref_mfcc = noisy.cuda(), ref_stft.cuda(), ref_mfcc.cuda()

    length = noisy.shape[1]

    idx = batch_match(noisy, ref_mfcc)
    noisy_stft = stft(noisy)
    with torch.no_grad():
        pre_spectra = model_1(noisy_stft, ref_stft, idx)

    enhanced_wav = istft(pre_spectra, length)
    idx = batch_match(enhanced_wav, ref_mfcc)
    with torch.no_grad():
        pre_spectra = model_2(noisy_stft, pre_spectra, ref_stft, idx)
    enhanced_wav = istft(pre_spectra, length)

    en = enhanced_wav.detach().squeeze().cpu().numpy()
    clean = clean.detach().squeeze().cpu().numpy()
    pesq_wb.append(pesq(fs=16000, ref=clean, deg=en, mode='wb'))

print(np.array(pesq_wb).mean())