from __future__ import print_function

import os
import numpy as np
import librosa
from glob import iglob

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


def log_mel_phase_to_signal(log_mel, phase):  # TODO
    return None
