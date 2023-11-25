"""Audio related classes & functions."""

import librosa
from scipy import signal
import numpy as np
import torch
from scipy.io.wavfile import read
import scipy
import pyworld as pw

from librosa.filters import mel as librosa_mel_fn


class Wav2Mel(object):

    def __init__(self, config):
        """init

        Args:
            config (str): audio config
        """
        self.config = config
        self.mel_basis = {}
        self.hann_window = {}

    def __call__(self, wav_fpath):
        """Wav to mel call logic.


        Args:
            wav_fpath (str): wav path

        Returns:
            mel(float32): mel bank
            wav(float32): wav sample
        """
        config = self.config
        wav, _ = librosa.load(wav_fpath, sr=config.sampling_rate)

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            config.sampling_rate,
            frame_period=config.hop_size / config.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch,
                             t, config.sampling_rate)
        pitch_length = wav.shape[0] // config.hop_size
        pitch = pitch[: pitch_length]
        if np.sum(pitch != 0) <= 1:
            raise ValueError("pitch should valid")

        wav = wav.astype(np.float32)

        mel, energy = self._torch_mel_spectrogram(
            torch.tensor(wav), config.fmin, config.fmax)
        energy = energy[: pitch_length]
        return mel, wav, pitch, energy

    def _load_wav_npy(self, npy_path):
        """load wav from npy

        Args:
            npy_path (str): wav path

        Returns:
            wav(float32): wav samples
        """
        wav = np.load(npy_path)
        wav = wav.astype(np.float32)
        return wav

    def _torch_mel_spectrogram(self, wav, fmin, fmax, center=False):
        """mel spectrogram torch version

        Args:
            wav (float tensor): wav sample
            fmin (int): frequency min 
            fmax (int): frequency max
            center (bool, optional): _description_. Defaults to False.

        Returns:
            spec(float32): mel bank
        """
        config = self.config
        if torch.min(wav) < -1.:
            print('min value is ', torch.min(wav))
        if torch.max(wav) > 1.:
            print('max value is ', torch.max(wav))

        if fmax not in self.mel_basis:
            mel = librosa_mel_fn(sr=config.sampling_rate,
                                 n_fft=config.fft_size, n_mels=config.num_mels, fmin=fmin, fmax=fmax)
            self.mel_basis[str(fmax) + '_' + str(wav.device)
                           ] = torch.from_numpy(mel).float().to(wav.device)
            self.hann_window[str(wav.device)] = torch.hann_window(
                config.win_size).to(wav.device)
        # pad data
        wav = torch.nn.functional.pad(wav, (int((config.fft_size - config.hop_size) / 2), int(
            (config.fft_size - config.hop_size) / 2)), mode='constant')

        spec = torch.stft(wav, n_fft=config.fft_size, hop_length=config.hop_size, win_length=config.win_size, pad_mode='constant',
                          window=self.hann_window[str(wav.device)], center=False, normalized=False, onesided=True, return_complex=False)
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        energy = torch.norm(spec, dim=0)
        spec = torch.matmul(
            self.mel_basis[str(fmax) + '_' + str(wav.device)], spec)
        spec = amp_to_db(spec) - self.config.ref_level_db
        spec = normalize_spec(spec, config.max_abs_value, config.min_level_db)
        return spec, energy


def amp_to_db(x, min_amp=1e-5):
    """spectrogram in amp domain to db domain

    Args:
        x (float): spectrogram
        min_amp (float, optional): min value. Defaults to 1e-5.

    Returns:
        （float tensor）: spectrogram in db domain
    """
    min_values = min_amp * torch.ones_like(x)
    return 20 * torch.log10(torch.maximum(min_values, x))


def normalize_spec(S, max_abs_value, min_level_db):
    """normalize spectrogram

    Args:
        S (float tensor): spectrogram
        max_abs_value (float): range
        min_level_db (int): min db

    Returns:
        (float tensor): normalized spectrogram
    """
    return torch.clip((2 * max_abs_value) * ((S - min_level_db)
                                             / (-min_level_db)) - max_abs_value,
                      -max_abs_value, max_abs_value)
