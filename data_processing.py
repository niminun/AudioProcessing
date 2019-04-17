from __future__ import print_function

import os
import numpy as np
import librosa
from glob import iglob

PHASE_EXT = "__phase.bin"
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
    if suffices is None:
        suffices = [""]

    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    for s in suffices:
        pattern = os.path.join(dir_path, "*{}".format(s))
        for f_path in iglob(pattern):

            # creating spectrogram
            stft = audio_file_to_stft(f_path)
            mel = spect_to_mel(np.abs(stft)**2, sr=SR, n_mels=N_MELS)
            log_mel = librosa.power_to_db(mel, ref=np.max)
            phase = np.angle(stft)

            # saving to disk
            f_name = os.path.splitext(os.path.basename(f_path))[0]
            log_mel.tofile(os.path.join(out_path, f_name + LOG_MEL_EXT))
            phase.tofile(os.path.join(out_path, f_name + PHASE_EXT))


def audio_file_to_stft(file_path, sr=SR, n_fft=2048, hop_length=None, win_length=None,
                       center=True):
    audio = AudioTrack(file_path, sr=sr)
    return librosa.core.stft(audio, n_fft=n_fft, hop_length=hop_length,
                             win_length=win_length, center=center)


def spect_to_mel(spect, sr=SR, n_fft=2048, hop_length=512, power=2.0, **kwargs):
    return librosa.feature.melspectrogram(sr=sr, S=spect, n_fft=n_fft,
                                          hop_length=hop_length, power=power,
                                          **kwargs)


def log_mel_phase_to_signal(log_mel, phase):  # TODO
    return None
