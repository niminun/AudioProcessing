from __future__ import print_function

import os
import random

import numpy as np
import librosa
from glob import iglob
from torch.utils.data import Dataset, DataLoader


PHASE_EXT = "__phase.bin"
LOG_MAG_EXT = "__log_mag.bin"
LOG_MEL_EXT = "__log_mel.bin"
FREQ_DIM = 0
TIME_DIM = 1
SR = 22050
N_MELS = 512


class AudioTrack(object):
    """
    A class for holding an audio object with the data required for analyzing it.
    """
    def __init__(self, path, sr=22050):
        self._signal, self._sr = librosa.load(path, sr)

    def signal(self):
        return self._signal

    def sr(self):
        return self._sr


def audio_dir_to_spects(dir_path, out_path, suffices=None):
    """
    Reading a directory of audio files, and saving for each the log magnitude,
    log mel spectrogram and phase.
    :param dir_path: path to a directory of audio files.
    :param out_path: path to an output directory.
    :param suffices: a list of relevant suffices.
    """
    if suffices is None:
        suffices = [""]

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    for s in suffices:
        pattern = os.path.join(dir_path, "*{}".format(s))
        for f_path in iglob(pattern):
            print(f_path)  # todo - change to log.
            # creating spectrogram
            stft = audio_file_to_stft(f_path)
            mag = np.abs(stft)
            phase = np.angle(stft)
            mel = spect_to_mel(mag**2, sr=SR, n_mels=N_MELS)

            # transforming spectrogram to log scale
            log_mag = librosa.amplitude_to_db(mag, ref=np.max)
            log_mel = librosa.power_to_db(mel, ref=np.max)

            # saving to disk
            f_name = os.path.splitext(os.path.basename(f_path))[0]
            log_mag.tofile(os.path.join(out_path, f_name + LOG_MAG_EXT))
            log_mel.tofile(os.path.join(out_path, f_name + LOG_MEL_EXT))
            phase.tofile(os.path.join(out_path, f_name + PHASE_EXT))


def audio_file_to_stft(file_path, sr=SR, n_fft=2048, hop_length=None, win_length=None,
                       center=True):
    audio = AudioTrack(file_path, sr=sr)
    return librosa.core.stft(audio.signal(), n_fft=n_fft, hop_length=hop_length,
                             win_length=win_length, center=center)


def spect_to_mel(spect, sr=SR, n_fft=2048, hop_length=512, power=2.0, **kwargs):
    return librosa.feature.melspectrogram(sr=sr, S=spect, n_fft=n_fft,
                                          hop_length=hop_length, power=power,
                                          **kwargs)


def log_mel_phase_to_signal(log_mel, phase):  # TODO - waiting for implementations in librosa
    return None


def log_mag_phase_to_signal(log_mag, phase):  # TODO - waiting for implementations in librosa
    return None


class SpectrogramDataset(Dataset):  # TODO - implement
    """Spectrogram dataset."""

    def __init__(self, in_dir, suffix, sample_width, spect_h, dtype='float32'):
        self._in_dir = in_dir
        self._suffix = suffix
        self._sample_width = sample_width
        self._spect_h = spect_h
        self._dtype = dtype
        self._paths = []
        self._avg_samp_per_spect = 0

        # calculate total number of samples in the data set.
        self._num_samples = 0
        self._calc_len()

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        # note that currently this gives the same probability to spects with different lengths.
        # consider adding some info about each spect length in the future s.t. a sample can be retrieved
        # with the true probability.
        spect_idx = int(idx / self._avg_samp_per_spect)
        spect = np.memmap(self._paths[spect_idx], dtype=self._dtype, mode='r').reshape(self._spect_h, -1)
        sample_start = random.randint(int(spect.shape[1]) - self._sample_width)
        return spect[sample_start: sample_start + self._sample_width]

    def _calc_len(self):
        self._num_samples = 0
        pattern = os.path.join(self._in_dir, "*{}".format(self._suffix))
        for f_path in iglob(pattern):
            spect = np.memmap(f_path, dtype=self._dtype, mode='r')
            width = len(spect) / float(self._spect_h)
            if not width.is_integer():
                print("file {} has problematic shape, skipping.".format(f_path))
                continue
            self._num_samples += int(width) - self._sample_width + 1
            self._paths.append(f_path)
        self._avg_samp_per_spect = self._num_samples / float(len(self._paths))
