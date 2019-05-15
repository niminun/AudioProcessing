from __future__ import print_function

import os
import random
import warnings

import numpy as np
import librosa
from glob import iglob
from torch.utils.data import Dataset, DataLoader


PHASE_EXT = "__phase.bin"
MAG_DB_EXT = "__mag_db.bin"
MEL_DB_EXT = "__mel_db.bin"
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
            # phase = np.angle(stft)
            # mel = spect_to_mel(mag**2, sr=SR, n_mels=N_MELS)

            # transforming spectrogram to log scale
            mag_db = librosa.amplitude_to_db(mag, top_db=120)
            # mel_db = librosa.power_to_db(mel, top_db=120)

            # saving to disk
            f_name = os.path.splitext(os.path.basename(f_path))[0]
            mag_db.tofile(os.path.join(out_path, f_name + MAG_DB_EXT))
            # log_mel.tofile(os.path.join(out_path, f_name + MEL_DB_EXT))
            # phase.tofile(os.path.join(out_path, f_name + PHASE_EXT))


def audio_file_to_stft(file_path, sr=SR, n_fft=2048, hop_length=None, win_length=None,
                       center=True):
    audio = AudioTrack(file_path, sr=sr)
    return librosa.core.stft(audio.signal(), n_fft=n_fft, hop_length=hop_length,
                             win_length=win_length, center=center)


def spect_to_mel(spect, sr=SR, n_fft=2048, hop_length=512, power=2.0, **kwargs):
    return librosa.feature.melspectrogram(sr=sr, S=spect, n_fft=n_fft,
                                          hop_length=hop_length, power=power,
                                          **kwargs)


def mel_db_to_signal(log_mel, phase):  # TODO - waiting for implementations in librosa
    return None


def mag_db_to_signal(mag_db, ref=1.0):
    mag = librosa.db_to_amplitude(mag_db, ref)
    return griffinlim(mag)


# todo - remove after version 0.7 of librosa
def griffinlim(S, n_iter=32, hop_length=None, win_length=None, window='hann',
               center=True, dtype=np.float32, length=None, pad_mode='reflect',
               momentum=0.99, random_state=None):

    """
    Approximate magnitude spectrogram inversion using the "fast" Griffin-Lim algorithm [1,2]_
    Given a short-time Fourier transform magnitude matrix (`S`), the algorithm randomly
    initializes phase estimates, and then alternates forward- and inverse-STFT
    operations.
    Note that this assumes reconstruction of a real-valued time-domain signal, and
    that `S` contains only the non-negative frequencies (as computed by
    `core.stft`).
    .. [1] Perraudin, N., Balazs, P., & Søndergaard, P. L.
        "A fast Griffin-Lim algorithm,"
        IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (pp. 1-4),
        Oct. 2013.
    .. [2] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.
    Parameters
    ----------
    S : np.ndarray [shape=(n_fft / 2 + 1, t), non-negative]
        An array of short-time Fourier transform magnitudes as produced by
        `core.stft`.
    n_iter : int > 0
        The number of iterations to run
    hop_length : None or int > 0
        The hop length of the STFT.  If not provided, it will default to `n_fft // 4`
    win_length : None or int > 0
        The window length of the STFT.  By default, it will equal `n_fft`
    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        A window specification as supported by `stft` or `istft`
    center : boolean
        If `True`, the STFT is assumed to use centered frames.
        If `False`, the STFT is assumed to use left-aligned frames.
    dtype : np.dtype
        Real numeric type for the time-domain signal.  Default is 32-bit float.
    length : None or int > 0
        If provided, the output `y` is zero-padded or clipped to exactly `length`
        samples.
    pad_mode : string
        If `center=True`, the padding mode to use at the edges of the signal.
        By default, STFT uses reflection padding.
    momentum : number >= 0
        The momentum parameter for fast Griffin-Lim.
        Setting this to 0 recovers the original Griffin-Lim method [1]_.
        Values near 1 can lead to faster convergence, but above 1 may not converge.
    random_state : None, int, or np.random.RandomState
        If int, random_state is the seed used by the random number generator
        for phase initialization.
        If `np.random.RandomState` instance, the random number generator itself;
        If `None`, defaults to the current `np.random` object.
    Returns
    -------
    y : np.ndarray [shape=(n,)]
        time-domain signal reconstructed from `S`
    See Also
    --------
    stft
    istft
    magphase
    filters.get_window
    """

    if random_state is None:
        rng = np.random
    elif isinstance(random_state, int):
        rng = np.random.RandomState(seed=random_state)
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state

    if momentum > 1:
        warnings.warn('Griffin-Lim with momentum={} > 1 can be unstable. Proceed with caution!'.format(momentum))
    elif momentum < 0:
        raise librosa.ParameterError('griffinlim() called with momentum={} < 0'.format(momentum))

    # Infer n_fft from the spectrogram shape
    n_fft = 2 * (S.shape[0] - 1)

    # randomly initialize the phase
    angles = np.exp(2j * np.pi * rng.rand(*S.shape))

    # And initialize the previous iterate to 0
    rebuilt = 0.

    for _ in range(n_iter):
        print(_)
        # Store the previous iterate
        tprev = rebuilt

        # Invert with our current estimate of the phases
        inverse = librosa.istft(S * angles, hop_length=hop_length, win_length=win_length,
                                window=window, center=center, dtype=dtype, length=length)

        # Rebuild the spectrogram
        rebuilt = librosa.stft(inverse, n_fft=n_fft, hop_length=hop_length,
                               win_length=win_length, window=window, center=center,
                               pad_mode=pad_mode)

        # Update our phase estimates
        angles[:] = rebuilt - (momentum / (1 + momentum)) * tprev
        angles[:] /= np.abs(angles) + 1e-16

    # Return the final phase estimates
    return librosa.istft(S * angles, hop_length=hop_length, win_length=win_length,
                         window=window, center=center, dtype=dtype, length=length)


class SpectrogramDataset(Dataset):
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
